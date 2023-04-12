#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench_dir='benchmarks/main/default'
test_dir='benchmarks/MPI/default'
n=4

mkdir -p $dir/$test_dir

mpirun -n $n ./vortex -d 0.0025 -t 0.0025