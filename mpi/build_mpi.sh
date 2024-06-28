#!/usr/bin/bash

target_file=$1

include=~/intel/oneapi/mpi/2021.11/include/
library=~/intel/oneapi/mpi/2021.11/lib/

export LD_LIBRARY_PATH=~/intel/oneapi/mpi/2021.11/lib:$LD_LIBRARY_PATH

g++ $target_file -o app -std=c++17 -I/usr/local/include/opencv4 -I$include -L/usr/local/lib -L$library -lOpenCL -lmpi

