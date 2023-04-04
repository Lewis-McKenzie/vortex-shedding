#! /usr/bin/bash
dir=$(pwd)

rm -r ./obj
make

bench='benchmarks/main/default'
test='benchmarks/OpenMP/default'

./vortex -d 0.0025 -o $test

python3 ./validation/validate.py $dir/$bench.vtk $dir/$test.vtk