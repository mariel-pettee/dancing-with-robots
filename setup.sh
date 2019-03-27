#!/usr/bin/env bash

# Path of the directory containing the script
export THIS_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Setting PATHs
export PATH="${THIS_DIRECTORY}/env/miniconda3/bin:$PATH" 

# Activate Conda environment
source activate aichoreo
