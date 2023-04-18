#! /usr/bin/bash
dir=$(pwd)

bench='benchmarks/main/default'

make clean
make

time nvprof ./vortex -d 0.0025 -x 256 - y 128
