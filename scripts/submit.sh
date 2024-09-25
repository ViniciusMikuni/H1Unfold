#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 18:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

#SBATCH --mail-user=ftoralesacosta@lbl.gov
#SBATCH --mail-type=ALL

export TF_CPP_MIN_LOG_LEVEL=2

#pretraining
# echo srun --mpi=pmi2 shifter python train.py --pretrain
# srun --mpi=pmi2 shifter python train.py --pretrain

#closure, load pretrain
echo srun --mpi=pmi2 shifter python train.py --load_pretrain --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure
srun --mpi=pmi2 shifter python train.py --load_pretrain --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure

# train from scratch, no loading pretrain or pretraining
# echo srun --mpi=pmi2 shifter python train.py --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure
# srun --mpi=pmi2 shifter python train.py --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure

#standard
# echo srun --mpi=pmi2 shifter python train.py
# srun --mpi=pmi2 shifter python train.py
