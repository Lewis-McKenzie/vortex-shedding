#! /usr/bin/bash
dir=$(pwd)

make clean
make

bench_dir='benchmarks/main/default'
test_dir='benchmarks/MPI/default'
n=4

mkdir -p $dir/$test_dir

f='-d 0.0025 --p -x 256 -y 128'

time mpirun -n $n ./vortex $f