#!/bin/bash
source /home/ucr/root_install/bin/thisroot.sh

input_file_name="dummy_input.root"
output_file_name="dummy_output.root"
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
echo ${input_file_name}
echo ${output_file_name}
echo ${jet_radius}
cp ./centauro.cxx ./centauro_${GPU_ID}.cxx
g++ centauro_${GPU_ID}.cxx -o centauro_${GPU_ID} `/home/ryan/fastjet-install/bin/fastjet-config --cxxflags --libs --plugins` `root-config --cflags --glibs` -lCentauro
echo ./centauro_${GPU_ID} --input "${input_file_name}" --output "${output_file_name}" --jet_radius "${jet_radius}"
./centauro_${GPU_ID} --input "${input_file_name}" --output "${output_file_name}" --jet_radius "${jet_radius}"
rm ./centauro_${GPU_ID} ./centauro_${GPU_ID}.cxx