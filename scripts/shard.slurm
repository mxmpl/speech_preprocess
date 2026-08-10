#!/bin/bash
#SBATCH --job-name=shard
#SBATCH --account=iyn@cpu
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --array=0-63

set -euo pipefail
speech-shard shard $SCRATCH/data/core/Libri-light/train-6000h $SCRATCH/data_sharded --size 1G --prefix Libri-light-train-6000h --resolve-symlinks --to-wav --extension .flac
