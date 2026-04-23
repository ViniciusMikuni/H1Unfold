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
```
### Saving unfolded weights to batch files

`save_unfolded_weights.py` reads a pre-evaluated weights file (produced by `evaluate.py`), clusters jets, computes observables, and writes the results to per-batch HDF5 files. Each MPI rank processes one independent batch of events in parallel, so the script scales to large datasets without loading everything into memory at once.

Each output file is named:
```
<base>_unfolded_<niter>[_reco][_boot]_batch<NNNN>.h5
```
and contains datasets: `jet_pt`, `jet_breit_pt`, `deltaphi`, `jet_tau10`, `zjet`, `zjet_breit`, plus `weights_nominal`, `weights<i>` (bootstrap), `mc_weights`, and `closure_weights` for MC files.

Start by initiating an interactive SLURM session:
```bash
salloc -C cpu -q interactive -t 240 -N 4 -A m3246 --image=vmikuni/tensorflow:ngc-23.12-tf2-v1 --cpus-per-task=64 --ntasks=16
```
Then,
```bash
srun --mpi=pmi2 shifter python save_unfolded_weights.py \
    --niter 4 \
    --load_pretrain \
    --file Rapgap_Eplus0607_prep.h5 \
    --nmax 18000000 \
    --batch_size 30000 \
    --data_folder /path/to/h5 \
    --pre_weights_file Rapgap_Eplus0607_unfolded_niter_4.h5 \
    --bootstrap \
    --nboot 50
```
Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_folder` | `/pscratch/sd/v/vmikuni/H1v2/h5` | Folder containing input and output h5 files |
| `--weights` | `../weights` | Folder with trained model weights |
| `--file` | `Rapgap_Eplus0607_prep.h5` | Input MC (or data) file |
| `--pre_weights_file` | `None` | HDF5 file with pre-evaluated unfolded weights from `evaluate.py` |
| `--niter` | `4` | OmniFold iteration to load |
| `--nmax` | `10000000` | Total number of events to process |
| `--batch_size` | `30000` | Events per rank/batch |
| `--bootstrap` | `False` | Save per-replica bootstrap weights |
| `--nboot` | `50` | Number of bootstrap replicas |
| `--reco` | `False` | Process reco-level events instead of gen-level |

## Plotting from batch files

`plot_from_batches.py` is a memory-efficient alternative to `plot.py`. Instead of loading all events at once, it iterates over the batch HDF5 files produced by `save_unfolded_weights.py` and accumulates weighted histograms for each observable.

Systematics, closure, and bootstrap statistical uncertainties are computed in the same way as the standard plotting code.

```bash
python plot_from_batches.py \
    --data_folder /path/to/h5 \
    --period Eplus0607 \
    --suffix boot \
    --niter 4 \
    --bootstrap \
    --nboot 50 \
    [--sys] [--reco] [--blind]
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_folder` | `/pscratch/sd/v/vmikuni/H1v2/h5` | Folder containing the batch h5 files |
| `--plot_folder` | `../plots` | Output directory for plots |
| `--period` | `Eplus0607` | Data-taking period string used in file names |
| `--suffix` | `boot` | Middle suffix in batch file names (e.g. `boot`) |
| `--niter` | `4` | OmniFold iteration to load |
| `--bootstrap` | `False` | Compute stat uncertainty from bootstrap replicas |
| `--nboot` | `50` | Number of bootstrap replicas |
| `--sys` | `False` | Load and propagate systematic variations |
| `--reco` | `False` | Plot reco-level results |
| `--blind` | `False` | Show closure results instead of data |

Plots are saved as PDFs named `<version>_<period>_<var>_<tag>.pdf` where `tag` is `unfolded`, `reco`, or `closure`.

<!-- ## Plotting

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
``` -->
