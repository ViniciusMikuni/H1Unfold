#!/bin/bash
SAMPLE=RapgapEp
PREFIX=Rapgap_Eplus0607
DATA_INPUT=/global/cfs/cdirs/m3246/H1/root/
DATA_OUTPUT=/global/cfs/cdirs/m3246/H1/h5
COMBINE=true # Combine the individual files after preparing and preprocessing
KEEP_INTERMEDIATE=false # Keep the individual files after preparing and preprocessing
NUM_PROCESSES=1 # Number of files to process in parallel

FLAGS="--num_processes $NUM_PROCESSES"
if [ "$COMBINE" = false ]; then FLAGS="$FLAGS --no-combine"; fi
if [ "$KEEP_INTERMEDIATE" = true ]; then FLAGS="$FLAGS --keep-intermediate"; fi

python prepare_data.py --sample $SAMPLE --data-input $DATA_INPUT --data-output $DATA_OUTPUT $FLAGS
python preprocess.py --file_name $PREFIX --data_folder $DATA_OUTPUT $FLAGS
