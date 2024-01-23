# H1 OmniFold

## Using shifter on Perlmutter
All the libraries required to run the code in the repo can be acessed through the docker image ```vmikuni/tensorflow:ngc-23.12-tf2-v1```. You can test it locally by doing:
```bash
shifter --image=vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18
python test_dataloader.py
```

You can also submit jobs to slurm using the same shifter image. For an interactive session you can do:
```bash
salloc -C gpu -q interactive  -t 10 -n 16 --ntasks-per-node=4 --gpus-per-task=1  -A m3246 --gpu-bind=none  --image vmikuni/tensorflow:ngc-23.04-tf2-v1 --module=gpu,nccl-2.15
```

For an example for a job submission file, look at ```submit.sh``` in the scripts folder.


## Training

Right now the closure test, standard unfolding and bootstrapping are available. Additional systematic uncertainties will be made available later as the respective files are processed. To train OmniFold you can run

```bash
python train.py [--closure] [--nstrap N]
```
