#!/bin/bash
#SBATCH --partition scavenge
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 1:00:00
#SBATCH --job-name red-rnn
#SBATCH --output logs/reduced-rnn-%J.log
#SBATCH -c 2

cd /project/hep/demers/mnp3/AI/dancing-with-robots/
source setup.sh
source activate env
python rnn_reduced_dim.py