#! /usr/bin/bash
dir=$(pwd)

make clean
make

n=16

export OMP_NUM_THREADS=$n

time ./vortex -d 0.0025 -x 256 -y 128 --p 