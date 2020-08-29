#!/bin/bash
#SBATCH -J TWEET
#SBATCH --output=t_sne.out
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
export BATCH_SIZE=36
export N_ITEMS=10000
export MPLBACKEND="agg"
python3 t_sne.py
