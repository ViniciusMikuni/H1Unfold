#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 00:30:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.04-tf2-v1
#SBATCH --module=gpu,nccl-2.15

export TF_CPP_MIN_LOG_LEVEL=2
echo srun --mpi=pmi2 shifter python train.py --closure
srun --mpi=pmi2 shifter python train.py --closure
