#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench='benchmarks/main/default'
test='benchmarks/CUDA/default'

mkdir $dir/$test

./vortex -d 0.0025 -o $test/vortex 2>&1 | tee $dir/$test/output.log

python3 ./validation/validate.py $dir/$bench.vtk $dir/$test/vortex.vtk