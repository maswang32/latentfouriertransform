#!/bin/bash
#SBATCH --nodelist=huang-l40s-1
#SBATCH --job-name=id_encoder 
#SBATCH --account=hi-res
#SBATCH --partition=hi-res
#SBATCH --qos=hi-res-main
#SBATCH --time=11:59:59
#SBATCH --output=id_encoder.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /data/scratch/ycda/gen/fmdiffae
/data/scratch/ycda/conda/envs/fmdiffae/bin/python train.py ckpt_path=/data/scratch/ycda/gen/fmdiffae/exp/runs/debug_run/2025-04-29/22-07-07/checkpoints/last.ckpt logger.id=mzkgum55