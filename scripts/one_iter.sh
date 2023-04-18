#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench_dir='benchmarks/main/once'
test_dir='benchmarks/MPI/once'
n=2

mkdir -p $dir/$test_dir

time mpirun -n $n ./vortex -d 0.0025 -t 0.0025 -o $test_dir/vortex 2>&1 | tee $dir/$test_dir/output.log

python3 ./validation/validate.py $dir/$bench_dir/vortex.vtk $dir/$test_dir/vortex.vtk