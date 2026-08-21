# H1 OmniFold

This branch is for comparing how different orders of combined Rapgap and Djangoh files impact the final trained OmniFold model.

## Processing the data
We combine the data in two ways: one with the original ordering that we used for the May 2025 model, and one in the file order obtained with `ls -lrth`. To prepare and process the data, please use `scripts/process_data.sh` with modified directories. Note that each of the two data directories will be about 525 GB (so a bit more than 1 TB in total).

To process the data, you'll need the tensorflow environment, `module load tensorflow`. I'm not sure if you need a CPU node or not but I've always used one for it (`salloc -C cpu -q interactive -t 240 -N 1 -A m3246`). This will take some time since it goes through each file one-by-one.

## Training the model
Now we train two different models with the two datasets. You can train with the script `scripts/submit.sh`. Each model training will take a bit under 4 hours with 5 iterations (set by default). You can either submit a job with `sbatch ./scripts/submit.sh`, or run one training at a time in an interactive GPU node (`salloc -C gpu -q interactive  -t 240 -n 16 --ntasks-per-node=4 --gpus-per-task=1  -A m3246 --gpu-bind=none  --image vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18`). Adjust the data paths in the script appropriately.

Note: Make sure you exit the tensorflow environment you used when processing the data. This applies to training the model and calculating the observables (below).
## Calculating and plotting the observables
Once the models are trained, you can then calculate and plot some jet observables. See `scripts/save_unfold.sh` for an example. I use a GPU node for this (80 minutes should be enough).
