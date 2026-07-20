# Speech dataset preprocessing

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
    --min-duration-off 2 \
    --max-duration-on 30.0
```

### `subsample`

Subsample turns of an RTTM file to match a target total duration following a
uniform distribution over `[min-duration, max-duration]` discretized in
`n-bins` bins.

```sh
speech-preprocess subsample PATH_RTTM PATH_SUBSAMPLED_RTTM \
    --target-hours 6000 \
    --min-duration 0.5 \
    --max-duration 30.0 \
    [--n-bins 1000]
```

### `segment`

Cut source audios into segments according to an RTTM file.

```sh
speech-preprocess segment PATH_AUDIOS PATH_RTTM PATH_OUTPUT \
    [--num-zeros 5] \
    [--extension .wav]
```

## API

<!-- griffe -->
## `speech_preprocess`

### `core`

#### `post_process_dataset`

```python
post_process_dataset(path_rttm, path_post_processed_rttm, *, min_duration_on, min_duration_off, max_duration_on, n_jobs=-1)
```

Post-process an RTTM file and write the result to a new file.

Applies three operations in order: merge consecutive speech segments separated by a silence
shorter than `min_duration_off`, discard segments shorter than `min_duration_on`, then
recursively split segments longer than `max_duration_on` using the longest silence in the
original annotation.

**Parameters:**

- **path_rttm** (<code>str | Path</code>) – Input RTTM file (or directory of `.rttm` files) to post-process.
- **path_post_processed_rttm** (<code>str | Path</code>) – Output RTTM file; results are appended to it.
- **min_duration_on** (<code>float</code>) – Discard speech segments shorter than this, in seconds.
- **min_duration_off** (<code>float</code>) – Merge consecutive segments separated by a silence shorter than this, in seconds.
- **max_duration_on** (<code>float</code>) – Split segments longer than this, in seconds.
- **n_jobs** (<code>int</code>) – Number of parallel jobs (passed to joblib). ``-1`` uses all CPUs.

#### `read_rttm`

```python
read_rttm(source)
```

Read an RTTM file into a Polars DataFrame.

**Parameters:**

- **source** (<code>str | Path | IO[str] | IO[bytes] | bytes</code>) – Path to the RTTM file, or a file-like object / raw bytes.

**Returns:**

- <code>DataFrame</code> – DataFrame with one row per turn and columns matching the ten RTTM fields:
- <code>DataFrame</code> – Type, File ID, Channel ID, Turn Onset, Turn Duration, Orthography Field,
- <code>DataFrame</code> – Speaker Type, Speaker Name, Confidence Score, Signal Lookahead Time.

#### `segment_dataset`

```python
segment_dataset(path_audios, path_rttm, path_output, *, num_zeros=5, extension='.wav', template='{uri}_{i:0{num_zeros}d}{extension}')
```

Cut source audios into segments according to an RTTM file.

For each turn in `path_rttm`, the corresponding slice of audio is extracted and written
to `path_output`. Output filenames are derived from `template`. Audio files with no turns
in the RTTM are returned as a list of unvoiced URIs.

**Parameters:**

- **path_audios** (<code>str</code>) – Directory containing the source audio files (16 kHz mono).
- **path_rttm** (<code>str</code>) – RTTM file describing the segments to extract.
- **path_output** (<code>str</code>) – Output directory for the extracted audio segments.
- **num_zeros** (<code>int</code>) – Zero-padding width for the segment index in output filenames.
- **extension** (<code>str</code>) – File extension of the source audios and output segments.
- **template** (<code>str</code>) – Filename template; receives ``uri``, ``i``, ``num_zeros``, and ``extension``.

**Returns:**

- <code>list[str]</code> – List of URIs for which no turns were found in the RTTM.

#### `subsample_dataset`

```python
subsample_dataset(path_rttm, path_subsampled_rttm, *, target_hours, min_duration, max_duration, density=lambda _: 1.0, n_bins=100)
```

Subsample turns from an RTTM file to reach a target total duration.

Turns are selected greedily to match the duration distribution defined by `density`
discretized into `n_bins` bins over `[min_duration, max_duration]`. Only turns whose
duration falls within that interval (inclusive) are eligible.

**Parameters:**

- **path_rttm** (<code>str | Path</code>) – Input RTTM file to subsample.
- **path_subsampled_rttm** (<code>str | Path</code>) – Output RTTM file with the selected turns.
- **target_hours** (<code>int</code>) – Target total duration of the subsampled set, in hours.
- **min_duration** (<code>float</code>) – Lower bound (inclusive) of the duration range, in seconds.
- **max_duration** (<code>float</code>) – Upper bound (inclusive) of the duration range, in seconds.
- **density** (<code>Callable[[float], float]</code>) – Probability density function over durations; defaults to uniform.
- **n_bins** (<code>int</code>) – Number of bins used to discretize the duration distribution.

#### `vad_dataset`

```python
vad_dataset(path_audios, path_rttm, *, model='pyannote/segmentation-3.0', token=None, extension='.wav')
```

Run voice activity detection over a directory of audios and append results to an RTTM file.

Detected speech turns are appended (not overwritten) to `path_rttm` using the pyannote
segmentation pipeline. Requires 16 kHz mono audio. Runs on GPU when available.

**Parameters:**

- **path_audios** (<code>str | Path</code>) – Directory containing the audio files to process.
- **path_rttm** (<code>str | Path</code>) – Output RTTM file; detected turns are appended to it.
- **model** (<code>str</code>) – Pretrained pyannote segmentation model identifier.
- **token** (<code>str | None</code>) – HuggingFace access token for gated models.
- **extension** (<code>str</code>) – File extension used to discover audios in `path_audios`.

#### `write_rttm`

```python
write_rttm(rttm, file)
```

Write a Polars DataFrame to an RTTM file.

**Parameters:**

- **rttm** (<code>DataFrame</code>) – DataFrame in the format returned by `read_rttm`.
- **file** (<code>str | Path | IO[str] | IO[bytes]</code>) – Destination path or file-like object.

### `post_process_dataset`

```python
post_process_dataset(path_rttm, path_post_processed_rttm, *, min_duration_on, min_duration_off, max_duration_on, n_jobs=-1)
```

Post-process an RTTM file and write the result to a new file.

Applies three operations in order: merge consecutive speech segments separated by a silence
shorter than `min_duration_off`, discard segments shorter than `min_duration_on`, then
recursively split segments longer than `max_duration_on` using the longest silence in the
original annotation.

**Parameters:**

- **path_rttm** (<code>str | Path</code>) – Input RTTM file (or directory of `.rttm` files) to post-process.
- **path_post_processed_rttm** (<code>str | Path</code>) – Output RTTM file; results are appended to it.
- **min_duration_on** (<code>float</code>) – Discard speech segments shorter than this, in seconds.
- **min_duration_off** (<code>float</code>) – Merge consecutive segments separated by a silence shorter than this, in seconds.
- **max_duration_on** (<code>float</code>) – Split segments longer than this, in seconds.
- **n_jobs** (<code>int</code>) – Number of parallel jobs (passed to joblib). ``-1`` uses all CPUs.

### `read_rttm`

```python
read_rttm(source)
```

Read an RTTM file into a Polars DataFrame.

**Parameters:**

- **source** (<code>str | Path | IO[str] | IO[bytes] | bytes</code>) – Path to the RTTM file, or a file-like object / raw bytes.

**Returns:**

- <code>DataFrame</code> – DataFrame with one row per turn and columns matching the ten RTTM fields:
- <code>DataFrame</code> – Type, File ID, Channel ID, Turn Onset, Turn Duration, Orthography Field,
- <code>DataFrame</code> – Speaker Type, Speaker Name, Confidence Score, Signal Lookahead Time.

### `segment_dataset`

```python
segment_dataset(path_audios, path_rttm, path_output, *, num_zeros=5, extension='.wav', template='{uri}_{i:0{num_zeros}d}{extension}')
```

Cut source audios into segments according to an RTTM file.

For each turn in `path_rttm`, the corresponding slice of audio is extracted and written
to `path_output`. Output filenames are derived from `template`. Audio files with no turns
in the RTTM are returned as a list of unvoiced URIs.

**Parameters:**

- **path_audios** (<code>str</code>) – Directory containing the source audio files (16 kHz mono).
- **path_rttm** (<code>str</code>) – RTTM file describing the segments to extract.
- **path_output** (<code>str</code>) – Output directory for the extracted audio segments.
- **num_zeros** (<code>int</code>) – Zero-padding width for the segment index in output filenames.
- **extension** (<code>str</code>) – File extension of the source audios and output segments.
- **template** (<code>str</code>) – Filename template; receives ``uri``, ``i``, ``num_zeros``, and ``extension``.

**Returns:**

- <code>list[str]</code> – List of URIs for which no turns were found in the RTTM.

### `subsample_dataset`

```python
subsample_dataset(path_rttm, path_subsampled_rttm, *, target_hours, min_duration, max_duration, density=lambda _: 1.0, n_bins=100)
```

Subsample turns from an RTTM file to reach a target total duration.

Turns are selected greedily to match the duration distribution defined by `density`
discretized into `n_bins` bins over `[min_duration, max_duration]`. Only turns whose
duration falls within that interval (inclusive) are eligible.

**Parameters:**

- **path_rttm** (<code>str | Path</code>) – Input RTTM file to subsample.
- **path_subsampled_rttm** (<code>str | Path</code>) – Output RTTM file with the selected turns.
- **target_hours** (<code>int</code>) – Target total duration of the subsampled set, in hours.
- **min_duration** (<code>float</code>) – Lower bound (inclusive) of the duration range, in seconds.
- **max_duration** (<code>float</code>) – Upper bound (inclusive) of the duration range, in seconds.
- **density** (<code>Callable[[float], float]</code>) – Probability density function over durations; defaults to uniform.
- **n_bins** (<code>int</code>) – Number of bins used to discretize the duration distribution.

### `vad_dataset`

```python
vad_dataset(path_audios, path_rttm, *, model='pyannote/segmentation-3.0', token=None, extension='.wav')
```

Run voice activity detection over a directory of audios and append results to an RTTM file.

Detected speech turns are appended (not overwritten) to `path_rttm` using the pyannote
segmentation pipeline. Requires 16 kHz mono audio. Runs on GPU when available.

**Parameters:**

- **path_audios** (<code>str | Path</code>) – Directory containing the audio files to process.
- **path_rttm** (<code>str | Path</code>) – Output RTTM file; detected turns are appended to it.
- **model** (<code>str</code>) – Pretrained pyannote segmentation model identifier.
- **token** (<code>str | None</code>) – HuggingFace access token for gated models.
- **extension** (<code>str</code>) – File extension used to discover audios in `path_audios`.

### `write_rttm`

```python
write_rttm(rttm, file)
```

Write a Polars DataFrame to an RTTM file.

**Parameters:**

- **rttm** (<code>DataFrame</code>) – DataFrame in the format returned by `read_rttm`.
- **file** (<code>str | Path | IO[str] | IO[bytes]</code>) – Destination path or file-like object.


<!-- /griffe -->
