#! /usr/bin/bash
dir=$(pwd)

make clean
make

./vortex -d 0.0025 -t 0.0025