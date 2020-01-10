#!/bin/bash
#
#SBATCH --job-name=ResNet-train
#SBATCH --output=ResNet-train.txt
#
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=5000

export PATH="/home/duchstf/miniconda3/bin:$PATH"
source activate RF-training
cd ~/home/duchstf/RF/hls4ml-RF/ResNet18
python3 RF-train-ResNet18.py
