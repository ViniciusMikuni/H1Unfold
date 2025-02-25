#!/bin/bash

input_file_name="dummy_input.h5"
output_file_name="dummy_output.h5"
jet_radius=1.0
GPU_ID=0

while [ True ]; do
if [ "$1" = "--input" ]; then
   input_file_name=$2
   shift 2 # past argument
elif [ "$1" = "--output" ]; then
   output_file_name=$2
   shift 2 # past argument
elif [ "$1" = "--jet_radius" ]; then
   jet_radius=$2
   shift 2 # past argument
elif [ "$1" = "--GPU_ID" ]; then
   GPU_ID=$2
   shift 2 # past argument
else
   break
fi
done
cp ./centauro.cxx ./centauro_${GPU_ID}.cxx
g++ centauro_${GPU_ID}.cxx -o centauro_${GPU_ID} \
    -I/global/common/software/nersc9/tensorflow/2.15.0/include/ \
    -L/global/common/software/nersc9/tensorflow/2.15.0/lib/ \
    `/global/cfs/cdirs/m3246/rmilton/fastjet-3.4.3-install/bin/fastjet-config --cxxflags --libs --plugins` \
    -lhdf5_cpp -lhdf5 -lCentauro
export LD_LIBRARY_PATH=/global/common/software/nersc9/tensorflow/2.15.0/lib:$LD_LIBRARY_PATH
./centauro_${GPU_ID} --input "${input_file_name}" --output "${output_file_name}" --jet_radius "${jet_radius}"
rm ./centauro_${GPU_ID} ./centauro_${GPU_ID}.cxx