#!/bin/bash
#SBATCH -J TWEET
#SBATCH --output=tweet.out
#SBATCH -t 72:00:00
#SBATCH --account=MST108431
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gp4d

module purge 
module load miniconda3
module load nvidia/cuda/10.1
module load cudnn/7.6
conda activate tf2

echo "JOB_ID: $SLURM_JOB_ID"

export BATCH_SIZE=36
python3 train.py