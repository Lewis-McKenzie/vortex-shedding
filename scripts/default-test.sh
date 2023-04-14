#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench='benchmarks/main/default'
test='benchmarks/CUDA/default'

mkdir -p $dir/$test

time ./vortex -d 0.0025 -o $test/vortex 2>&1 | tee $dir/$test/output.log

python3 ./validation/validate.py $dir/$bench/vortex.vtk $dir/$test/vortex.vtk