import copy
from collections import defaultdict
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal, overload

import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment, Timeline
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["post_process_dataset", "read_rttm", "segment_dataset", "subsample_dataset", "vad_dataset", "write_rttm"]


# ------------
# 0. Utilities
# ------------


SAMPLE_RATE: int = 16_000


class AudioDataset(Dataset):
    def __init__(self, root: str | Path, *, extension: str) -> None:
        assert extension[0] == ".", "Extension should start with a '.', like in '.wav'"
        self.paths = sorted(Path(root).rglob(f"*{extension}"))

    def __getitem__(self, index: int) -> tuple[str, AudioDecoder]:
        path = self.paths[index].resolve()
        decoder = AudioDecoder(path)
        assert decoder.metadata.sample_rate == SAMPLE_RATE, f"{path.stem} has {decoder.metadata.sample_rate=}"
        assert decoder.metadata.num_channels == 1, f"{path.stem} is not mono"
        return path.stem, decoder

    def __len__(self) -> int:
        return len(self.paths)


def read_rttm(source: str | Path | IO[str] | IO[bytes] | bytes) -> pl.DataFrame:
    return pl.read_csv(
        source,
        has_header=False,
        new_columns=[
            "Type",
            "File ID",
            "Channel ID",
            "Turn Onset",
            "Turn Duration",
            "Orthography Field",
            "Speaker Type",
            "Speaker Name",
            "Confidence Score",
            "Signal Lookahead Time",
        ],
        separator=" ",
        schema_overrides={
            "Type": pl.String,
            "File ID": pl.String,
            "Turn Onset": pl.Float64,
            "Turn Duration": pl.Float64,
            "Speaker Name": pl.String,
        },
        null_values="<NA>",
    )


def write_rttm(rttm: pl.DataFrame, file: str | Path | IO[str] | IO[bytes]) -> None:
    rttm.write_csv(file, separator=" ", null_value="<NA>", include_header=False, float_precision=3)


# ---------------------------
# 1. Voice Activity Detection
# ---------------------------


def vad_dataset(
    path_audios: str | Path,
    path_rttm: str | Path,
    *,
    model: str = "pyannote/segmentation-3.0",
    token: str | None = None,
    extension: str = ".wav",
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    path_rttm = Path(path_rttm)
    dataset = AudioDataset(path_audios, extension=extension)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation = Model.from_pretrained(model, token=token)
    pipeline = VoiceActivityDetection(segmentation=segmentation).to(device).eval()  # ty: ignore[invalid-argument-type]
    pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    for uri, decoder in (pbar := tqdm(dataset)):
        pbar.set_description(uri)
        waveform = decoder.get_all_samples().data.to(device)
        vad = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad.uri = uri
        with path_rttm.open("a", encoding="utf-8") as f:
            vad.write_rttm(f)


# ------------
# 2. Filtering
# ------------

# ------------------
# 3. Post-processing
# ------------------


def rttm_to_annotations(rttm: pl.DataFrame, *, target_type: str = "SPEAKER", n_jobs: int = -1) -> list[Annotation]:
    def aux(uri: str, turns: pl.DataFrame) -> Annotation:
        annotation = Annotation(uri=uri)
        for i, turn in enumerate(turns.iter_rows(named=True)):
            if turn["Type"] != target_type:
                continue
            segment = Segment(turn["Turn Onset"], turn["Turn Onset"] + turn["Turn Duration"])
            annotation[segment, i] = turn["Speaker Name"]
        return annotation

    annotations = Parallel(n_jobs=n_jobs)(delayed(aux)(uri, turns) for (uri,), turns in rttm.group_by("File ID"))
    return sorted(annotations, key=lambda x: x.uri)


def read_annotations(path: str | Path) -> list[Annotation]:
    rttm = read_rttm(path) if Path(path).is_file() else pl.concat([read_rttm(p) for p in Path(path).glob("*.rttm")])
    return rttm_to_annotations(rttm)


def map_segment_to_silences(original: Annotation, processed: Annotation) -> dict[Segment, Timeline]:
    """Maps each speech segment in the processed annotation to the silences
    occurring within the corresponding segment in the original annotation.
    """
    mapping = defaultdict(list)
    silences_in_processed = processed.extrude(Timeline(original.itersegments()))
    for (seg, _), (silence, _) in processed.co_iter(silences_in_processed):
        mapping[seg].append(silence)
    return {seg: Timeline(mapping[seg]) for seg in processed.itersegments()}


def split_by_silence(
    segment: Segment,
    silences: Timeline,
    min_duration_on: float | None,
    max_duration_on: float,
) -> Timeline:
    """Recursively split `segment` using the longest silence in `silences` until
    all resulting segments are shorter than `max_duration_on`. Segments shorter than
    `min_duration_on` are discarded."""
    timeline = Timeline([segment])
    if min_duration_on is not None and timeline.duration() <= min_duration_on:  # Remove too short segments
        return Timeline([])
    if timeline.duration() <= max_duration_on:  # Termination condition: the segment is short enough
        return timeline
    if not silences:  # No more silence to split the segment: cut in the middle
        left = split_by_silence(Segment(segment.start, segment.middle), silences, min_duration_on, max_duration_on)
        right = split_by_silence(Segment(segment.middle, segment.end), silences, min_duration_on, max_duration_on)
        return Timeline(left.segments_list_ + right.segments_list_)
    longest_hole = max(silences, key=lambda x: x.duration)  # Find the longest silence
    timeline = timeline.extrude(longest_hole)
    new_timelines = [split_by_silence(seg, silences.crop(seg), min_duration_on, max_duration_on) for seg in timeline]  # ty: ignore[invalid-argument-type]
    return Timeline([s for t in new_timelines for s in t.segments_list_])


def cut_long_segments(
    original: Annotation,
    active: Annotation,
    min_duration_on: float | None,
    max_duration_on: float,
) -> Annotation:
    new_segments, segment_to_silences = [], map_segment_to_silences(original, active)
    for seg in active.itersegments():
        silences = segment_to_silences[seg]
        new_segments += list(split_by_silence(seg, silences, min_duration_on, max_duration_on).segments_list_)
    annotation = Annotation(uri=original.uri)
    for i, segment in enumerate(new_segments):
        annotation[segment, i] = "SPEECH"
    return annotation


def remove_long_silences(annotation: Annotation, min_duration_off: float) -> Annotation:
    return annotation.support(collar=min_duration_off)


def remove_short_segments(annotation: Annotation, min_duration_on: float) -> Annotation:
    active = copy.deepcopy(annotation)
    for segment, track, *_ in list(active.itertracks()):
        if segment.duration < min_duration_on:
            del active[segment, track]
    return active


def post_process_annotation(
    original: Annotation,
    *,
    min_duration_on: float,
    min_duration_off: float,
    max_duration_on: float,
) -> Annotation:
    active = copy.deepcopy(original)
    if min_duration_off > 0.0:
        active = remove_long_silences(active, min_duration_off)
    if min_duration_on > 0.0:
        active = remove_short_segments(active, min_duration_on)
    if max_duration_on > 0.0:
        active = cut_long_segments(original, active, min_duration_on, max_duration_on)
    return active


def post_process_dataset(
    path_rttm: str | Path,
    path_post_processed_rttm: str | Path,
    *,
    min_duration_on: float,
    min_duration_off: float,
    max_duration_on: float,
    n_jobs: int = -1,
) -> None:
    annotations = Parallel(n_jobs=n_jobs)(
        delayed(post_process_annotation)(
            annotation,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            max_duration_on=max_duration_on,
        )
        for annotation in read_annotations(path_rttm)
    )
    for annotation in annotations:
        with Path(path_post_processed_rttm).open("a", encoding="utf-8") as f:
            annotation.write_rttm(f)


# --------------
# 4. Subsampling
# --------------


@overload
def subsample(
    array: Iterable[float],
    target: int,
    *,
    lower: float,
    upper: float,
    density: Callable[[float], float],
    n_bins: int,
    return_counts: Literal[False] = False,
) -> list[int]: ...


@overload
def subsample(
    array: Iterable[float],
    target: int,
    *,
    lower: float,
    upper: float,
    density: Callable[[float], float],
    n_bins: int,
    return_counts: Literal[True] = True,
) -> tuple[list[int], np.ndarray]: ...


def subsample(
    array: Iterable[float],
    target: int,
    *,
    lower: float,
    upper: float,
    density: Callable[[float], float],
    n_bins: int,
    return_counts: bool = False,
) -> list[int] | tuple[list[int], np.ndarray]:
    bin_width = (upper - lower) / n_bins
    centers = [lower + (i + 0.5) * bin_width for i in range(n_bins)]
    weights = [max(density(c), 0.0) for c in centers]
    norm = sum(w * c for w, c in zip(weights, centers, strict=True))
    if norm <= 0:
        raise ValueError("density integrates to zero over [lower, upper]")
    scale = target / norm
    bin_counts, per_bin_count = [0] * n_bins, [round(w * scale) for w in weights]
    indices = []
    for i, value in enumerate(array):
        if value < lower or value > upper:
            continue
        bin_idx = min(int((value - lower) / bin_width), n_bins - 1)
        if bin_counts[bin_idx] < per_bin_count[bin_idx]:
            bin_counts[bin_idx] += 1
            indices.append(i)
    if return_counts:
        return indices, np.vstack((bin_counts, per_bin_count))
    return indices


def subsample_dataset(
    path_rttm: str | Path,
    path_subsampled_rttm: str | Path,
    *,
    target_hours: int,
    min_duration: float,
    max_duration: float,
    density: Callable[[float], float] = lambda _: 1.0,
    n_bins: int = 100,
) -> None:
    rttm = read_rttm(path_rttm)
    indices = subsample(
        rttm["Turn Duration"].to_list(),
        target_hours * 3600,
        lower=min_duration,
        upper=max_duration,
        n_bins=n_bins,
        density=density,
    )
    write_rttm(rttm[indices], path_subsampled_rttm)


# ---------------
# 5. Segmentation
# ---------------


def segment_dataset(
    path_audios: str,
    path_rttm: str,
    path_output: str,
    *,
    num_zeros: int = 5,
    extension: str = ".wav",
    template: str = "{uri}_{i:0{num_zeros}d}{extension}",
) -> list[str]:
    output = Path(path_output)
    unvoiced = []
    rttm = read_rttm(path_rttm).with_columns((pl.col("Turn Onset") + pl.col("Turn Duration")).alias("Turn Offset"))
    dataset = AudioDataset(path_audios, extension=extension)
    for uri, decoder in tqdm(dataset):
        segments = rttm.filter(pl.col("File ID") == uri)
        if segments.height == 0:
            unvoiced.append(uri)
        for i, (onset, offset) in enumerate(segments.sort("Turn Onset")[["Turn Onset", "Turn Offset"]].iter_rows()):
            dest = output / template.format(uri=uri, i=i, num_zeros=num_zeros, extension=extension)
            dest.parent.mkdir(parents=True, exist_ok=True)
            samples = decoder.get_samples_played_in_range(onset, offset)
            AudioEncoder(samples.data, sample_rate=samples.sample_rate).to_file(dest)
    return unvoiced
