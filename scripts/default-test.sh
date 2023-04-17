#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench_dir='benchmarks/main/default'
test_dir='benchmarks/MPI/default'
n=4

mkdir -p $dir/$test_dir

f='-d 0.0025 --p'

time mpirun -n $n ./vortex $f -o $test_dir/vortex 2>&1 | tee $dir/$test_dir/output.log

python3 ./validation/validate.py $dir/$bench_dir/vortex.vtk $dir/$test_dir/vortex.vtk