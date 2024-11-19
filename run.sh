#!/bin/bash
prog="./ljp"
infile="config.in"

# find any .in file and set it as the input file
if [ -f *.in ]; then
    infile=$(ls *.in)
fi

mkdir -p output/cpu
mkdir -p output/cuda

series="20 40 80 100"

# read -c for cpu, -g for gpu, otherwise both
if [ "$1" == "-c" ]; then
    rm -rf output/cpu/*
    # run cpu version
    echo "----------------- CPU $size * $size mols -------------------"
    for size in $series; do
        # create sub folder
        mkdir -p output/cpu/$size
        rm -rf output/cpu/$size/*

        # run cpu version
        echo -n "Running CPU version($size x $size)"
        $prog $infile $size 0 > "output/cpu/$size/final"
        echo " - $(grep '^\[CPU Time\]' output/cpu/$size/final)"
    done
elif [ "$1" == "-g" ]; then
    rm -rf output/cuda/*
    # run gpu version
    echo "----------------- CUDA -------------------"
    for size in $series; do
        # create sub folder
        mkdir -p output/cuda/$size
        rm -rf output/cuda/$size/*
        # run cpu version
        echo -n "Running CUDA version($size x $size)"
        $prog $infile $size 1 > "output/cuda/$size/final"
        echo " - $(grep '^\[GPU Time\]' output/cuda/$size/final)"
    done
else
    rm -rf output/cpu/*
    rm -rf output/cuda/*
   # run cpu version
    echo "----------------- CPU -------------------"
    for size in $series; do
        # create sub folder
        mkdir -p output/cpu/$size
        rm -rf output/cpu/$size/*
        # run cpu version
        echo -n "Running CPU version($size x $size)"
        $prog $infile $size 0 0 > "output/cpu/$size/final"
        echo " - $(grep '^\[CPU Time\]' output/cpu/$size/final)"
    done

    # run gpu version
    echo "----------------- CUDA -------------------"
    for size in $series; do
        # create sub folder
        mkdir -p output/cuda/$size
        rm -rf output/cuda/$size/*
        # run cpu version
        echo -n "Running CUDA version($size x $size)"
        $prog $infile $size 1 0 > "output/cuda/$size/final"
        echo " - $(grep '^\[GPU Time\]' output/cuda/$size/final)"
    done
fi