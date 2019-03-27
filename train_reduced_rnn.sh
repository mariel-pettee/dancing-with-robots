#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 10:00:00
#SBATCH --job-name pca-gpu
#SBATCH --output logs/pca-gpu-%J.log
#SBATCH -c 2

cd /project/hep/demers/mnp3/AI/dancing-with-robots/
source setup.sh
python python/pca_to_chor-rnn.py