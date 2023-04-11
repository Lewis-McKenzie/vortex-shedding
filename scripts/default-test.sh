#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench='benchmarks/main/default'
test='benchmarks/CUDA/default'

./vortex -d 0.0025 -o $test 2>&1 | tee $dir/$test_dir/output.log

python3 ./validation/validate.py $dir/$bench.vtk $dir/$test.vtk