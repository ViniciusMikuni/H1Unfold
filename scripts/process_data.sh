#!/bin/bash

input_dir=/global/cfs/cdirs/m3246/H1/root/
final_dir=/pscratch/sd/r/rmilton/H1_filechecking_originalordering/

mkdir -p $final_dir
# Eplus samples

echo python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "RapgapEp"
python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "RapgapEp"
echo python preprocess.py --data_folder $final_dir --file_name "Rapgap_Eplus0607.h5"
python preprocess.py --data_folder $final_dir --file_name "Rapgap_Eplus0607.h5"

echo python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "DjangohEp"
python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "DjangohEp"
echo python preprocess.py --data_folder $final_dir --file_name "Djangoh_Eplus0607.h5"
python preprocess.py --data_folder $final_dir --file_name "Djangoh_Eplus0607.h5"


final_dir=/pscratch/sd/r/rmilton/H1_filechecking_stringordering/

mkdir -p $final_dir
echo python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "RapgapEp" --string_ordering
python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "RapgapEp" --string_ordering
echo python preprocess.py --data_folder $final_dir --file_name "Rapgap_Eplus0607.h5"
python preprocess.py --data_folder $final_dir --file_name "Rapgap_Eplus0607.h5"

echo python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "DjangohEp" --string_ordering
python prepare_data.py --data-input $input_dir --data-output $final_dir --sample "DjangohEp" --string_ordering
echo python preprocess.py --data_folder $final_dir --file_name "Djangoh_Eplus0607.h5"
python preprocess.py --data_folder $final_dir --file_name "Djangoh_Eplus0607.h5"