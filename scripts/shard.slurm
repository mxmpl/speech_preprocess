#!/bin/bash
#SBATCH --job-name=shard
#SBATCH --account=iyn@cpu
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --array=0-63

set -euo pipefail
speech-shard shard $SCRATCH/data/$4/$1/$2 $SCRATCH/data_sharded/$4/$1/ --size 1G --prefix $1-$2 --resolve-symlinks --to-wav --extension $3
