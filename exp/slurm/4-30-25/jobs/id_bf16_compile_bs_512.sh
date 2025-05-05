#!/bin/bash
#SBATCH --nodelist=huang-l40s-1
#SBATCH --job-name=id_bf16_compile_bs_512
#SBATCH --account=hi-res
#SBATCH --partition=hi-res
#SBATCH --qos=hi-res-main
#SBATCH --time=23:59:59
#SBATCH --output=../logs/id_bf16_compile_bs_512.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /data/scratch/ycda/gen/fmdiffae
/data/scratch/ycda/conda/envs/fmdiffae/bin/python train.py trainer.precision="bf16" compile=True name=id_bf16_compile_bs_512 data.batch_size=512