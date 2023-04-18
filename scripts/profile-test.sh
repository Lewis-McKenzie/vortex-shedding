#! /usr/bin/bash
dir=$(pwd)

bench='benchmarks/main/default'

make clean
make

nvprof ./vortex -d 0.0025

python3 ./validation/validate.py $dir/$bench/vortex.vtk $dir/out/vortex.vtk