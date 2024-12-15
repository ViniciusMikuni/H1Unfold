#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 38:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

#SBATCH --mail-user=ftoralesacosta@lbl.gov
#SBATCH --mail-type=ALL

export TF_CPP_MIN_LOG_LEVEL=2

# train from scratch, no loading pretrain or  running pretraining
echo srun --mpi=pmi2 shifter python train.py --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure config_Dec_lrscale5_patience10.json
srun --mpi=pmi2 shifter python train.py --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure config_Dec_lrscale5_patience10.json

#standard
# echo srun --mpi=pmi2 shifter python train.py
# srun --mpi=pmi2 shifter python train.py
