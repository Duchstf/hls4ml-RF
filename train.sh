#!/bin/bash
#
#SBATCH --job-name=RF-train
#SBATCH --output=RF-train.txt
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=5000

export PATH="/home/duchstf/miniconda3/bin:$PATH"
source activate RF-training
cd ~/RF/hls4ml-RF/
python3 RF-prune-CNN-95.py



