#! /usr/bin/bash

make clean
make

nvprof ./vortex -d 0.0025 -t 0.25
