#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 07:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

export TF_CPP_MIN_LOG_LEVEL=2

#pretraining
echo srun --mpi=pmi2 shifter python train.py --pretrain
srun --mpi=pmi2 shifter python train.py --pretrain

#closure
# echo srun --mpi=pmi2 shifter python train.py --load_pretrain --closure
# srun --mpi=pmi2 shifter python train.py --load_pretrain --closure

#standard
# echo srun --mpi=pmi2 shifter python train.py
# srun --mpi=pmi2 shifter python train.py
