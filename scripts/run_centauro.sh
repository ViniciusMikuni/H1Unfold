#!/bin/bash
module load python
conda activate cernroot
g++ centauro.cxx -o centauro `/global/homes/r/rmilton/m3246/rmilton/fastjet-install/bin/fastjet-config --cxxflags --libs --plugins` `root-config --cflags --glibs` -lCentauro
./centauro