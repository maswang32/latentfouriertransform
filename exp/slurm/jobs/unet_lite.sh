#!/bin/bash
#SBATCH --nodelist=huang-l40s-1
#SBATCH --job-name=unet_lite
#SBATCH --account=hi-res
#SBATCH --partition=hi-res
#SBATCH --qos=hi-res-main
#SBATCH --time=23:59:59
#SBATCH --output=../logs/unet_lite.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /data/scratch/ycda/gen/fmdiffae
/data/scratch/ycda/conda/envs/fmdiffae/bin/python train.py model=fftmask_unet_lite name=unet_lite ckpt_path=/data/scratch/ycda/gen/fmdiffae/exp/runs/unet_lite/2025-04-30/13-32-50/checkpoints/last.ckpt logger.id=klbbesy0
