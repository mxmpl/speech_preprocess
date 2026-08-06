#!/bin/bash
#SBATCH --job-name=shard
#SBATCH --account=iyn@cpu
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --array=0-511

set -euo pipefail
speech-shard manifest $1 $SCRATCH/data_sharded --verify
