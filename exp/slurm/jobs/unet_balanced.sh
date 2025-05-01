#!/bin/bash
#SBATCH --nodelist=huang-l40s-1
#SBATCH --job-name=unet_balanced
#SBATCH --account=hi-res
#SBATCH --partition=hi-res
#SBATCH --qos=hi-res-main
#SBATCH --time=23:59:59
#SBATCH --output=../logs/unet_balanced.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /data/scratch/ycda/gen/fmdiffae
/data/scratch/ycda/conda/envs/fmdiffae/bin/python train.py model=fftmask_unet_balanced name=unet_balanced trainer.precision="bf16" compile=True
