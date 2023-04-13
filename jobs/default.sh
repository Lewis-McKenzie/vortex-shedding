#!/bin/bash
#SBATCH --time=00:10:00                 # Maximum time (HH:MM:SS)
#SBATCH --ntasks=1                      # run on a single CPU
#SBATCH --mem=1gb                       # reserve 1GB memory for job
#SBATCH --output=default%j.log          # standard output and error log
#SBATCH --partition=teach               # run in the teaching queue
 
echo default running on `hostname`

bench='benchmarks/main/default'
test='benchmarks/CUDA/default'

mkdir -p $dir/$test

./vortex -d 0.0025 -o $test/vortex 2>&1 | tee $dir/$test/output.log

python3 ./validation/validate.py $dir/$bench/vortex.vtk $dir/$test/vortex.vtk