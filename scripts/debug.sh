#! /usr/bin/bash
dir=$(pwd)

make clean
make

n=6
export OMP_NUM_THREADS=$n

./vortex -d 0.0025 -t 0.0025