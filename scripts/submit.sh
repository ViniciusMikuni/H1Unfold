#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 16
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 08:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

export TF_CPP_MIN_LOG_LEVEL=2

# Old ordering of files
echo srun --mpi=pmi2 shifter python train.py --load_pretrain --closure --data_folder /pscratch/sd/r/rmilton/H1_filechecking_originalordering/ --config config_general.json
srun --mpi=pmi2 shifter python train.py --load_pretrain --closure --data_folder /pscratch/sd/r/rmilton/H1_filechecking_originalordering/ --config config_general.json

# String ordering of files
echo srun --mpi=pmi2 shifter python train.py --load_pretrain --closure --data_folder /pscratch/sd/r/rmilton/H1_filechecking_stringordering/ --config config_general_stringorder.json
srun --mpi=pmi2 shifter python train.py --load_pretrain --closure --data_folder /pscratch/sd/r/rmilton/H1_filechecking_stringordering/ --config config_general_stringorder.json