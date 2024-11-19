#!/bin/bash
prog="./ljp"
infile="config.in"

# find any .in file and set it as the input file
if [ -f *.in ]; then
    infile=$(ls *.in)
fi

mkdir -p output/cpu
mkdir -p output/cuda

series="10 20 40 80 100 200 400"
verbose=0

if [ "$1" == "-v" ]; then
    verbose=1
fi

# read -c for cpu, -g for gpu, otherwise both
if [ "$1" == "-c" ]; then
    # run cpu version
    echo "----------------- CPU $size * $size mols -------------------"
    for size in $series; do
        # run cpu version
        echo -n "Running CPU version($size x $size)"
        $prog $infile $size 0 > "output/cpu/$size/final"
        echo " - $(grep '^\[CPU Time\]' output/cpu/$size/final)"
    done
elif [ "$1" == "-g" ]; then
    echo "----------------- CUDA -------------------"
    for size in $series; do
        # run cpu version
        echo -n "Running CUDA version($size x $size)"
        $prog $infile $size 1 > "output/cuda/$size/final"
        echo " - $(grep '^\[GPU Time\]' output/cuda/$size/final)"
    done
else
   # run cpu version
    echo "----------------- CPU -------------------"
    for size in $series; do
        # run cpu version
        echo -n "Running CPU version($size x $size)"
        $prog $infile $size 0 $verbose > "output/cpu/$size/final"
        echo " - $(grep '^\[CPU Time\]' output/cpu/$size/final)"
    done
    # run gpu version
    echo "----------------- CUDA -------------------"
    for size in $series; do
        # run cpu version
        echo -n "Running CUDA version($size x $size)"
        $prog $infile $size 1 $verbose > "output/cuda/$size/final"
        echo " - $(grep '^\[GPU Time\]' output/cuda/$size/final)"
    done
fi