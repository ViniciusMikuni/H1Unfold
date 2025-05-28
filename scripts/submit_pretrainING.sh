#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 2:00:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

#SBATCH --mail-user=ftoralesacosta@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --array=0-60
#SBATCH --mail-type=ALL

export TF_CPP_MIN_LOG_LEVEL=2

echo "srun --mpi=pmi2 shifter python train.py --load_pretrain --closure --jobNum $SLURM_ARRAY_TASK_ID"
srun --mpi=pmi2 shifter python train.py --load_pretrain --closure --jobNum $SLURM_ARRAY_TASK_ID
