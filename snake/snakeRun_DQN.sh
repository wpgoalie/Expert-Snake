#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=04:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=8GB            ## Allocated Memory
#SBATCH --cpus-per-task 16    ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs

rm -r snake-agent
python snakeAgent_DQN.py
