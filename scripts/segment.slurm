#!/bin/bash
#SBATCH --job-name=segment
#SBATCH --account=iyn@cpu
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --array=0-255

set -euo pipefail
speech-preprocess segment $DSDIR/Libri-light/large $SCRATCH/speech_preprocess/libri-light-large-complement-6000k.rttm $SCRATCH/data/core/Libri-light/train-6000h --extension .flac
