#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 3:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

#SBATCH --mail-user=twamorkar@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --array=0-50

export TF_CPP_MIN_LOG_LEVEL=2

echo "srun --mpi=pmi2 shifter python train.py --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure --config config_March_lrscale5_patience10_parallel.json $SLURM_ARRAY_TASK_ID"
srun --mpi=pmi2 shifter python train.py --data_folder /global/cfs/cdirs/m3246/vmikuni/H1v2/h5/ --closure --config config_March_lrscale5_patience10_parallel.json --jobNum $SLURM_ARRAY_TASK_ID
#standard
# echo srun --mpi=pmi2 shifter python train.py
# srun --mpi=pmi2 shifter python train.py
