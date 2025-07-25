# H1 OmniFold

## Contributing

### Linting
To lint the code, run:

```bash
ruff format .
ruff check --fix .
```

## Packages

You can run the code using the docker container with instructions written below or use the NERSC tensorflow module.

### Using the module and installing additional packages

Besides the tensorflow module, we only require a few additional packages that you can install with pip after loading the module. You can use the commands:

```bash
module load tensorflow
pip install uproot awkward fastjet
```

### Preparing new datasets
Assuming you have root files coming from the outputs of the H1 analysis code, you need to:
- convert the files to HDF5.

```bash
cd scripts
python prepare_data.py --sample DjangohEp --data-input INPUT/LOCATION --data-output OUTPUT/LOCATION
```
- preprocess the inputs to convert them into the format expected by the model.

```bash
python preprocess.py --file_name DjangohEplus0607.h5
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

### Model Pre-Training

You can create the pretrained model using the following commands after creating the datasets:

```bash
python train.py --pretrain
```

Use the config files under the JSON folder ```config_general.json``` and ```config_omnifold.json``` to determine the baseline parameters of the training.

### Unfolding

Right now the closure test, standard unfolding and bootstrapping are available. Additional systematic uncertainties will be made available later as the respective files are processed. To train OmniFold you can run

```bash
python train.py [--closure] [--nstrap N] [--dataset ep/em]
```

### Evaluating the unfolding results

After the unfolding is done, you can run the evaluation code to load the trained network, determine the unfolded weights, and then save the results to a new file containing the same information as the original file on top of the new weights

```bash
python evaluate.py [--dataset ep/em] [--niter I] [--load_pretrain] [--file Rapgap] [--bootstrap]


## Plotting

Plotting can be performed using either multiple GPUs or in a single GPU. The script to run is:

```bash
module load tensorflow
python plot.py --niter 4 [--closure] [--load_pretrain] [--reco]
```

The ```--reco``` flag is used to load reco level events to verify the reweighting of step 1. The ```--closure``` flag is used to load the training using the closure test. The ```--load_pretrain``` flag uses the training that starts from a pre-trained model

To plot using multiple GPUs run:

```bash
salloc -C gpu -q interactive  -t 40 -n 16 --ntasks-per-node=4 --gpus-per-task=1  -A m3246  --gpu-bind None
module load tensorflow
[pip install --user fastjet] #Do it only once
srun  python plot.py --niter 5 --closure --load_pretrain --data_folder /global/cfs/cdirs/m3246/H1/h5/ --weights /global/cfs/cdirs/m3246/H1/weights/
```
