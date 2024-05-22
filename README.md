# H1 OmniFold

## Packages

You can run the code using the docker container with instructions written below or use the NERSC tensorflow module.

### Using the module and installing additional packages

Besides the tensorflow module, we only require a few additional packages that you can install with pip after loading the module. You can use the commands:

```bash
module load tensorflow
pip install uproot
```

### Using shifter on Perlmutter
All the libraries required to run the code in the repo can be acessed through the docker image ```vmikuni/tensorflow:ngc-23.12-tf2-v1```. You can test it locally by doing:
```bash
shifter --image=vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18
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

## Plotting

Plotting can be performed using either multiple GPUs or in a single GPU. The script to run is:

```bash
python plot.py --niter 5 [--closure] [--load_pretrain] [--reco]
```

The ```--reco``` flag is used to load reco level events to verify the reweighting of step 1. The ```--closure``` flag is used to load the training using the closure test. The ```--load_pretrain``` flag uses the training that starts from a pre-trained model
