#!/bin/bash
module load python
conda activate cernroot

input_file_name="dummy_input.root"
output_file_name="dummy_output.root"
while [ True ]; do
if [ "$1" = "--input" ]; then
   input_file_name=$2
   shift 2 # past argument
elif [ "$1" = "--output" ]; then
   output_file_name=$2
   shift 2 # past argument
else
   break
fi
done
echo ${input_file_name}
echo ${output_file_name}
g++ centauro.cxx -o centauro `/global/homes/r/rmilton/m3246/rmilton/fastjet-install/bin/fastjet-config --cxxflags --libs --plugins` `root-config --cflags --glibs` -lCentauro
./centauro --input "${input_file_name}" --output "${output_file_name}"