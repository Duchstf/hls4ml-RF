#!/bin/bash
#
#SBATCH --job-name=RF-train
#SBATCH --output=RF-train.txt
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=5000

module load miniconda

source activate mnist-training
cd ~/RF/hls4ml-RF/
python3 RF-prune-CNN.py



