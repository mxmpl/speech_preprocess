# preprocess

A small toolkit for preparing speech datasets: voice activity detection, RTTM
post-processing, duration-based subsampling, and audio segmentation.

## Installation

```sh
uv sync
```

Requires Python 3.12+.

## Library

The exported functions live in `speech_preprocess.core`:

- `read_rttm` — read an RTTM file into a Polars DataFrame.
- `write_rttm` — write a Polars DataFrame back to an RTTM file.
- `vad_dataset` — run pyannote voice activity detection over a directory of audios.
- `post_process_dataset` — clean up an RTTM file (drop short turns, merge over short silences, split long turns).
- `subsample_dataset` — subsample turns of an RTTM file to match a target duration distribution.
- `segment_dataset` — extract audio chunks from source files according to an RTTM file.

```python
from speech_preprocess import (
    post_process_dataset,
    read_rttm,
    segment_dataset,
    subsample_dataset,
    vad_dataset,
    write_rttm,
)
```

## CLI

The package ships a CLI with one subcommand per exported function:

```sh
speech-preprocess <command> [options]
```

Run `speech-preprocess <command> --help` for the full list of options.

### `vad`

Run voice activity detection over a directory of audios and append the detected
turns to an RTTM file.

```sh
speech-preprocess vad PATH_AUDIOS PATH_RTTM \
    [--model pyannote/segmentation-3.0] \
    [--token HF_TOKEN] \
    [--extension .wav]
```

### `post-process`

Post-process an RTTM file: discard short speech segments, merge over short
silences, and split overly long segments using the longest silence in the
original annotation.

```sh
speech-preprocess post-process PATH_RTTM PATH_POST_PROCESSED_RTTM \
    --min-duration-on 0.5 \
    --min-duration-off 0.2 \
    --max-duration-on 30.0
```

### `subsample`

Subsample turns of an RTTM file to match a target total duration following a
uniform distribution over `[min-duration, max-duration]` discretized in
`n-bins` bins.

```sh
speech-preprocess subsample PATH_RTTM PATH_SUBSAMPLED_RTTM \
    --target-hours 100 \
    --min-duration 1.0 \
    --max-duration 30.0 \
    [--n-bins 100]
```

### `segment`

Cut source audios into segments according to an RTTM file.

```sh
speech-preprocess segment PATH_AUDIOS PATH_RTTM PATH_OUTPUT \
    [--num-zeros 5] \
    [--extension .wav]
```

## API

<!-- pydoc-markdown -->
### read\_rttm

```python
def read_rttm(
        source: str | Path | IO[str] | IO[bytes] | bytes) -> pl.DataFrame
```

Read an RTTM file into a Polars DataFrame.

**Arguments**:

- `source` - Path to the RTTM file, or a file-like object / raw bytes.
  

**Returns**:

  DataFrame with one row per turn and columns matching the ten RTTM fields:
  Type, File ID, Channel ID, Turn Onset, Turn Duration, Orthography Field,
  Speaker Type, Speaker Name, Confidence Score, Signal Lookahead Time.

### write\_rttm

```python
def write_rttm(rttm: pl.DataFrame,
               file: str | Path | IO[str] | IO[bytes]) -> None
```

Write a Polars DataFrame to an RTTM file.

**Arguments**:

- `rttm` - DataFrame in the format returned by `read_rttm`.
- `file` - Destination path or file-like object.

### vad\_dataset

```python
def vad_dataset(path_audios: str | Path,
                path_rttm: str | Path,
                *,
                model: str = "pyannote/segmentation-3.0",
                token: str | None = None,
                extension: str = ".wav") -> None
```

Run voice activity detection over a directory of audios and append results to an RTTM file.

Detected speech turns are appended (not overwritten) to `path_rttm` using the pyannote
segmentation pipeline. Requires 16 kHz mono audio. Runs on GPU when available.

**Arguments**:

- `path_audios` - Directory containing the audio files to process.
- `path_rttm` - Output RTTM file; detected turns are appended to it.
- `model` - Pretrained pyannote segmentation model identifier.
- `token` - HuggingFace access token for gated models.
- `extension` - File extension used to discover audios in `path_audios`.

### map\_segment\_to\_silences

```python
def map_segment_to_silences(original: Annotation,
                            processed: Annotation) -> dict[Segment, Timeline]
```

Maps each speech segment in the processed annotation to the silences
occurring within the corresponding segment in the original annotation.

### split\_by\_silence

```python
def split_by_silence(segment: Segment, silences: Timeline,
                     min_duration_on: float | None,
                     max_duration_on: float) -> Timeline
```

Recursively split `segment` using the longest silence in `silences` until
all resulting segments are shorter than `max_duration_on`. Segments shorter than
`min_duration_on` are discarded.

### post\_process\_dataset

```python
def post_process_dataset(path_rttm: str | Path,
                         path_post_processed_rttm: str | Path,
                         *,
                         min_duration_on: float,
                         min_duration_off: float,
                         max_duration_on: float,
                         n_jobs: int = -1) -> None
```

Post-process an RTTM file and write the result to a new file.

Applies three operations in order: merge consecutive speech segments separated by a silence
shorter than `min_duration_off`, discard segments shorter than `min_duration_on`, then
recursively split segments longer than `max_duration_on` using the longest silence in the
original annotation.

**Arguments**:

- `path_rttm` - Input RTTM file (or directory of `.rttm` files) to post-process.
- `path_post_processed_rttm` - Output RTTM file; results are appended to it.
- `min_duration_on` - Discard speech segments shorter than this, in seconds.
- `min_duration_off` - Merge consecutive segments separated by a silence shorter than this, in seconds.
- `max_duration_on` - Split segments longer than this, in seconds.
- `n_jobs` - Number of parallel jobs (passed to joblib). ``-1`` uses all CPUs.

### subsample\_dataset

```python
def subsample_dataset(path_rttm: str | Path,
                      path_subsampled_rttm: str | Path,
                      *,
                      target_hours: int,
                      min_duration: float,
                      max_duration: float,
                      density: Callable[[float], float] = lambda _: 1.0,
                      n_bins: int = 100) -> None
```

Subsample turns from an RTTM file to reach a target total duration.

Turns are selected greedily to match the duration distribution defined by `density`
discretized into `n_bins` bins over `[min_duration, max_duration]`. Only turns whose
duration falls within that interval (inclusive) are eligible.

**Arguments**:

- `path_rttm` - Input RTTM file to subsample.
- `path_subsampled_rttm` - Output RTTM file with the selected turns.
- `target_hours` - Target total duration of the subsampled set, in hours.
- `min_duration` - Lower bound (inclusive) of the duration range, in seconds.
- `max_duration` - Upper bound (inclusive) of the duration range, in seconds.
- `density` - Probability density function over durations; defaults to uniform.
- `n_bins` - Number of bins used to discretize the duration distribution.

### segment\_dataset

```python
def segment_dataset(
        path_audios: str,
        path_rttm: str,
        path_output: str,
        *,
        num_zeros: int = 5,
        extension: str = ".wav",
        template: str = "{uri}_{i:0{num_zeros}d}{extension}") -> list[str]
```

Cut source audios into segments according to an RTTM file.

For each turn in `path_rttm`, the corresponding slice of audio is extracted and written
to `path_output`. Output filenames are derived from `template`. Audio files with no turns
in the RTTM are returned as a list of unvoiced URIs.

**Arguments**:

- `path_audios` - Directory containing the source audio files (16 kHz mono).
- `path_rttm` - RTTM file describing the segments to extract.
- `path_output` - Output directory for the extracted audio segments.
- `num_zeros` - Zero-padding width for the segment index in output filenames.
- `extension` - File extension of the source audios and output segments.
- `template` - Filename template; receives ``uri``, ``i``, ``num_zeros``, and ``extension``.
  

**Returns**:

  List of URIs for which no turns were found in the RTTM.


<!-- /pydoc-markdown -->
