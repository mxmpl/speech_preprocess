# preprocess

A small toolkit for preparing speech datasets: voice activity detection, RTTM
post-processing, duration-based subsampling, and audio segmentation.

## Installation

```sh
uv sync
```

Requires Python 3.14+.

## Library

The exported functions live in `preprocess.core`:

- `vad_dataset` — run pyannote voice activity detection over a directory of audios.
- `post_process_dataset` — clean up an RTTM file (drop short turns, merge over short silences, split long turns).
- `subsample_dataset` — subsample turns of an RTTM file to match a target duration distribution.
- `segment_dataset` — extract audio chunks from source files according to an RTTM file.

```python
from preprocess import vad_dataset, post_process_dataset, subsample_dataset, segment_dataset
```

## CLI

The package ships a CLI with one subcommand per exported function:

```sh
python -m preprocess <command> [options]
```

Run `python -m preprocess <command> --help` for the full list of options.

### `vad`

Run voice activity detection over a directory of audios and append the detected
turns to an RTTM file.

```sh
python -m preprocess vad PATH_AUDIOS PATH_RTTM \
    [--model pyannote/segmentation-3.0] \
    [--token HF_TOKEN] \
    [--extension .wav]
```

### `post-process`

Post-process an RTTM file: discard short speech segments, merge over short
silences, and split overly long segments using the longest silence in the
original annotation.

```sh
python -m preprocess post-process PATH_RTTM PATH_POST_PROCESSED_RTTM \
    --min-duration-on 0.5 \
    --min-duration-off 0.2 \
    --max-duration-on 30.0
```

### `subsample`

Subsample turns of an RTTM file to match a target total duration following a
uniform distribution over `[min-duration, max-duration]` discretized in
`n-bins` bins.

```sh
python -m preprocess subsample PATH_RTTM PATH_SUBSAMPLED_RTTM \
    --target-hours 100 \
    --min-duration 1.0 \
    --max-duration 30.0 \
    [--n-bins 100]
```

### `segment`

Cut source audios into segments according to an RTTM file.

```sh
python -m preprocess segment PATH_AUDIOS PATH_RTTM PATH_OUTPUT \
    [--num-zeros 5] \
    [--extension .wav] \
```
