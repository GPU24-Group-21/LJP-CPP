#!/bin/bash
prog="./ljp"
infile="config.in"

# find any .in file and set it as the input file
if [ -f *.in ]; then
    infile=$(ls *.in)
fi

mkdir -p output/cpu
mkdir -p output/cuda

series="10 20 40"
verbose=0
mode="a"

while getopts "cgv" opt; do
    case $opt in
        c)
            mode="c"
            ;;
        g)
            mode="g"
            ;;
        v)
            verbose=1
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# read -c for cpu, -g for gpu, otherwise both
if [ $mode == "c" ]; then
    # run cpu version
    echo "----------------- CPU -------------------"
    for size in $series; do
        mkdir -p output/cpu/$size
        rm -f output/cpu/$size/*
        # run cpu version
        echo -n "Running CPU version($size x $size)"
        $prog $infile $size 0 $verbose > "output/cpu/$size/final"
        echo " - $(grep '^\[CPU Time\]' output/cpu/$size/final)"

        if [ $verbose == 1 ]; then
            python3 plot.py output/cpu/$size &
        fi
    done
elif [ $mode == "g" ]; then
    echo "----------------- CUDA -------------------"
    for size in $series; do
        mkdir -p output/cuda/$size
        rm -f output/cuda/$size/*
        # run cpu version
        echo -n "Running CUDA version($size x $size)"
        $prog $infile $size 1 $verbose > "output/cuda/$size/final"
        echo " - $(grep '^\[GPU Time\]' output/cuda/$size/final)"

        if [ $verbose == 1 ]; then
            python3 plot.py output/cuda/$size &
        fi
    done
else
   # run cpu version
    echo "----------------- CPU -------------------"
    for size in $series; do
        mkdir -p output/cpu/$size
        rm -f output/cpu/$size/*
        # run cpu version
        echo -n "Running CPU version($size x $size)"
        $prog $infile $size 0 $verbose > "output/cpu/$size/final"
        echo " - $(grep '^\[CPU Time\]' output/cpu/$size/final)"

        if [ $verbose == 1 ]; then
            python3 plot.py output/cpu/$size &
        fi
    done
    # run gpu version
    echo "----------------- CUDA -------------------"
    for size in $series; do
        mkdir -p output/cuda/$size
        rm -f output/cuda/$size/*
        # run cpu version
        echo -n "Running CUDA version($size x $size)"
        $prog $infile $size 1 $verbose > "output/cuda/$size/final"
        echo " - $(grep '^\[GPU Time\]' output/cuda/$size/final)"

        if [ $verbose == 1 ]; then
            python3 plot.py output/cuda/$size &
        fi
    done
fi