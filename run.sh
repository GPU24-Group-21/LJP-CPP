#!/bin/bash
prog="./ljp"
infile="Rap_2_LJP.in"

# find any .in file and set it as the input file
if [ -f *.in ]; then
    infile=$(ls *.in)
fi

mkdir -p output/cpu
mkdir -p output/cuda

mkdir -p timing/cpu
mkdir -p timing/cuda

# clear the output folder
rm -rf output/cpu/*
rm -rf output/cuda/*

clear

series="10 20 40"

# run cpu version
for size in $series; do
    # create sub folder
    mkdir -p output/cpu/$size
    rm -rf output/cpu/$size/*
    # run cpu version
    echo "Running CPU version($size x $size)"
    $prog $infile $size 1 0 > "output/cpu/$size/final"
done

# run gpu version
echo "\n------------------------------------"
for size in $series; do
    # create sub folder
    mkdir -p output/cuda/$size
    rm -rf output/cuda/$size/*
    # run cpu version
    echo "Running CUDA version($size x $size)"
    $prog $infile $size 1 1 > "output/cuda/$size/final"
done