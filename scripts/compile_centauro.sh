#!/bin/bash
# Compile centauro.cxx into ./centauro
# Usage: ./compile_centauro.sh [path/to/fastjet-config]
fastjet_config_path=${1:-../fastjet-3.4.3-install/bin/fastjet-config}
tf_dir=/global/common/software/nersc9/tensorflow/2.15.0

# rpath lets ./centauro find the hdf5 libraries without setting LD_LIBRARY_PATH
g++ centauro.cxx -o centauro \
    -I${tf_dir}/include/ \
    -L${tf_dir}/lib/ -Wl,-rpath,${tf_dir}/lib \
    `${fastjet_config_path} --cxxflags --libs --plugins` \
    -lhdf5_cpp -lhdf5 -lCentauro
