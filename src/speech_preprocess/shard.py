"""Shard a dataset of audio files into tar archives and build a JSONL manifest.

Two stages, both distributed across SLURM tasks/array jobs when available:

1. ``shard_dataset`` walks a (possibly nested) directory of audio files and packs them into
   uncompressed tar archives, rolling over to a new archive whenever the cap ``size`` (in bytes)
   would be exceeded. Files can optionally be transcoded to WAV on the fly.
2. ``write_manifest`` reads those archives back and emits a JSONL manifest with one record per
   audio file, giving its location inside the archive so the bytes can be memory-mapped later.
"""

import argparse
import io
import mmap
import os
import tarfile
from pathlib import Path

import polars as pl
from filelock import FileLock
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

from .core import SAMPLE_RATE, split_for_distributed

__all__ = ["reindex_shards", "shard_dataset", "write_manifest"]

MANIFEST_FIELDS = ["fileid", "path", "num_samples", "archive", "byte_offset", "byte_size"]


def worker_id() -> int:
    """Return a unique index for the current SLURM worker (``0`` when not distributed).

    Mirrors the array-then-rank splitting done by ``split_for_distributed`` so that each worker
    owns a distinct slice of the file list and can name its outputs without colliding.
    """
    if "SLURM_NTASKS" not in os.environ:
        return 0
    rank, world_size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    return array_id * world_size + rank


class CompressedArchiveError(Exception):
    """Archive must be uncompressed to be read by offset."""


class NoAudioFileError(Exception):
    """No audio file found in the archives."""

    def __init__(self, source: Path | str, extension: str) -> None:
        super().__init__(f"No file with extension `{extension}` found in {source}.")


class ConversionError(Exception):
    """Failed to transcode an audio file to WAV."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"Failed to convert {path} to WAV.")


# ---------
# Utilities
# ---------


def parse_size(value: str) -> int:
    """Parse a human-readable byte size such as ``512M``, ``1.5G`` or ``2048`` into bytes."""
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    text = value.strip().upper().removesuffix("B")
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def num_samples(source: str | Path | bytes, *, verify: bool = False) -> int:
    """Return the number of samples of an audio file, read from its header."""
    metadata = AudioDecoder(source).metadata
    samples = metadata.duration_seconds_from_header * metadata.sample_rate
    if verify and not samples.is_integer():
        where = f" in {source}" if isinstance(source, (str, Path)) else ""
        raise ValueError(f"Number of samples {samples} is not an integer{where}")
    return int(samples)


def bytes_from_archive(archive: Path | str, offset: int, size: int) -> bytes:
    """Read ``size`` bytes at ``offset`` from ``archive`` via memory mapping."""
    with Path(archive).open("rb") as file, mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
        return mapped[offset : offset + size]


def _to_wav_bytes(path: Path) -> bytes:
    samples = AudioDecoder(path).get_all_samples()
    encoder = AudioEncoder(samples.data, sample_rate=samples.sample_rate)
    return bytes(encoder.to_tensor("wav", sample_rate=SAMPLE_RATE, num_channels=1))


# ---------
# Sharding
# ---------


class ShardWriter:
    """Pack files into a sequence of uncompressed tar archives, capped at ``max_size`` bytes.

    Archives are named ``{prefix}-{worker:05d}-{index:05d}.tar`` so that concurrent workers never
    collide. A single file larger than ``max_size`` is still written to its own archive.
    """

    def __init__(self, output: Path, prefix: str, worker: int, max_size: int) -> None:
        self.output = output
        self.prefix = prefix
        self.worker = worker
        self.max_size = max_size
        self.index = 0
        self.current_size = 0
        self.tar: tarfile.TarFile | None = None

    def _open(self) -> None:
        self.tar = tarfile.open(self.output / f"{self.prefix}-{self.worker:05d}-{self.index:05d}.tar", "w")  # ruff:ignore[open-file-with-context-handler]
        self.current_size = 0

    def _reserve(self, member_size: int) -> None:
        if self.tar is None:
            self._open()
        elif self.current_size > 0 and self.current_size + member_size > self.max_size:
            self.tar.close()
            self.index += 1
            self._open()

    def add_file(self, path: Path, arcname: str) -> None:
        size = path.stat().st_size
        self._reserve(size)
        assert self.tar is not None
        self.tar.add(path, arcname=arcname)
        self.current_size += size

    def add_bytes(self, data: bytes, arcname: str) -> None:
        self._reserve(len(data))
        assert self.tar is not None
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))
        self.current_size += len(data)

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def shard_dataset(
    dataset: str | Path,
    output: str | Path,
    *,
    size: int,
    extension: str = ".wav",
    to_wav: bool = False,
    prefix: str = "shard",
) -> None:
    """Pack a directory of audio files into uncompressed tar archives.

    Files are discovered recursively, sorted, and split across SLURM workers when running in a
    distributed setting. Each archive is closed once adding the next file would exceed ``size``.
    The relative path of every file is preserved as its name inside the archive.

    Args:
        dataset: Directory containing the audio files (may be nested).
        output: Directory where the tar archives are written; created if missing.
        size: Maximum archive size, in bytes.
        extension: File extension used to discover audios in ``dataset``.
        to_wav: Transcode each file to WAV before archiving (renames the member to ``.wav``).
        prefix: Prefix for the generated archive filenames.
    """
    root = Path(dataset).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = split_for_distributed(sorted(root.rglob(f"*{extension}")))
    writer = ShardWriter(output, prefix, worker_id(), size)
    for path in tqdm(paths):
        relative = path.relative_to(root)
        if not to_wav:
            writer.add_file(path, str(relative))
            continue
        try:
            data = _to_wav_bytes(path)
        except (ValueError, RuntimeError) as error:
            writer.close()  # Flush the shards written so far before crashing on a bad input.
            raise ConversionError(path) from error
        writer.add_bytes(data, str(relative.with_suffix(".wav")))
    writer.close()


def reindex_shards(output: str | Path, *, prefix: str = "shard") -> list[Path]:
    """Collapse the per-worker shard names into a single, contiguous hexadecimal index.

    ``shard_dataset`` names archives ``{prefix}-{worker:05d}-{shard:05d}.tar`` so that concurrent
    workers never collide. Once every worker has finished, this renames them to
    ``{prefix}-{index:0Nx}.tar`` with one global index (sorted by their original name), so the
    archive set is numbered contiguously. Run once, from a single process, after the distributed
    sharding step.

    Args:
        output: Directory containing the tar archives produced by ``shard_dataset``.
        prefix: Prefix of the archive filenames.

    Returns:
        The new archive paths, in index order.
    """
    output = Path(output)
    shards = sorted(output.glob(f"{prefix}-*-*.tar"))
    width = len(f"{max(len(shards) - 1, 0):x}")
    renamed = []
    for index, shard in enumerate(tqdm(shards)):
        dest = output / f"{prefix}-{index:0{width}x}.tar"
        shard.rename(dest)
        renamed.append(dest)
    return renamed


# ---------
# Manifest
# ---------


def _archive_paths(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_dir():
        return sorted(source.rglob("*.tar"))
    if tarfile.is_tarfile(source):
        return [source]
    raise ValueError(f"{source} is neither a directory nor a tar file.")


def _manifest_records(archive: Path, extension: str, *, verify: bool) -> list[dict[str, object]]:
    with tarfile.open(archive, "r") as tar:
        if tar.mode != "r":
            raise CompressedArchiveError
        members = tar.getmembers()
    return [
        {
            "fileid": Path(info.name).stem,
            "path": info.name,
            "num_samples": num_samples(bytes_from_archive(archive, info.offset_data, info.size), verify=verify),
            "archive": str(archive),
            "byte_offset": info.offset_data,
            "byte_size": info.size,
        }
        for info in members
        if not info.isdir() and info.name.endswith(extension)
    ]


def write_manifest(
    archives: str | Path,
    output: str | Path,
    *,
    extension: str = ".wav",
    verify: bool = False,
) -> None:
    """Build a JSONL manifest from a directory of tar archives (or a single archive).

    Each record locates one audio file by archive, byte offset and byte size, alongside its number
    of samples, so the raw bytes can later be memory-mapped without unpacking the archive. Archives
    are split across SLURM workers when distributed; every worker appends its records to the single
    ``output`` file, serialising the appends with a file lock.

    Args:
        archives: Directory containing the tar archives, or a single tar file.
        output: Path to the output ``.jsonl`` manifest; records are appended to it.
        extension: Only members with this extension are included.
        verify: Check that every decoded sample count is an integer.
    """
    paths = split_for_distributed(_archive_paths(Path(archives)))
    records = [record for archive in tqdm(paths) for record in _manifest_records(archive, extension, verify=verify)]
    if not records:
        raise NoAudioFileError(archives, extension)
    output = Path(output)
    content = pl.from_dicts(records).select(MANIFEST_FIELDS).write_ndjson()
    with FileLock(f"{output}.lock"), output.open("a", encoding="utf-8") as file:
        file.write(content)


# ---------
# Entrypoint
# ---------


def cli() -> None:
    parser = argparse.ArgumentParser(prog="speech-shard", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_shard = subparsers.add_parser("shard", help="Pack audio files into capped tar archives")
    p_shard.add_argument("dataset", help="Directory containing the audio files to shard")
    p_shard.add_argument("output", help="Output directory for the tar archives")
    p_shard.add_argument(
        "--size",
        type=parse_size,
        required=True,
        help="Maximum archive size, e.g. 512M, 1.5G or a raw byte count",
    )
    p_shard.add_argument("--extension", default=".wav", help="Extension used to find audios in dataset")
    p_shard.add_argument("--to-wav", action="store_true", help="Transcode files to WAV before archiving")
    p_shard.add_argument("--prefix", default="shard", help="Prefix for the generated archive filenames")

    p_re = subparsers.add_parser("reindex", help="Collapse per-worker shards into a single hex index")
    p_re.add_argument("output", help="Directory containing the tar archives to reindex")
    p_re.add_argument("--prefix", default="shard", help="Prefix of the archive filenames")

    p_man = subparsers.add_parser("manifest", help="Build a JSONL manifest from tar archives")
    p_man.add_argument("archives", help="Directory of tar archives, or a single tar file")
    p_man.add_argument("output", help="Output .jsonl manifest path")
    p_man.add_argument("--extension", default=".wav", help="Only include members with this extension")
    p_man.add_argument("--verify", action="store_true", help="Check that every sample count is an integer")

    args = parser.parse_args()
    if args.command == "shard":
        shard_dataset(
            args.dataset,
            args.output,
            size=args.size,
            extension=args.extension,
            to_wav=args.to_wav,
            prefix=args.prefix,
        )
    elif args.command == "reindex":
        reindex_shards(args.output, prefix=args.prefix)
    elif args.command == "manifest":
        write_manifest(args.archives, args.output, extension=args.extension, verify=args.verify)
    else:
        parser.error("Invalid subcommand")


if __name__ == "__main__":
    cli()
