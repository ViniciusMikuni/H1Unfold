# H1 OmniFold

## Slurm jobs with shifter 
salloc -C gpu -q interactive  -t 10 -n 16 --ntasks-per-node=4 --gpus-per-task=1  -A m3246 --gpu-bind=none  --image vmikuni/tensorflow:ngc-23.04-tf2-v1 --module=gpu,nccl-2.15

