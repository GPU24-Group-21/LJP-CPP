#!/bin/bash

output_folder="output"
if [ ! -d $output_folder ]; then
    echo "Output folder does not exist"
    exit 1
fi

if [ ! -d "$output_folder/cpu" ] || [ ! -d "$output_folder/cuda" ]; then
    echo "Output folder does not contain the cpu and cuda subfolders"
    exit 1
fi

for cpu_folder in $output_folder/cpu/*; do
    folder_name=$(basename $cpu_folder)
    cuda_folder="${cpu_folder/cpu/cuda}"
    if [ ! -d "$cuda_folder" ]; then
        echo "Missing corresponding CUDA folder for $cpu_folder"
        exit 1
    fi
    if [ ! -f "$cpu_folder/final" ] || [ ! -f "$cuda_folder/final" ]; then
        echo "Missing final output file in $cpu_folder or $cuda_folder"
        exit 1
    fi
    for cpu_file in $cpu_folder/*.out; do
        cuda_file="${cpu_file/cpu/cuda}"
        if [ ! -f "$cuda_file" ]; then
            echo "Missing corresponding CUDA file for $cpu_file"
            exit 1
        fi
        
        ./validator $cpu_file $cuda_file
        
        if [ $? -ne 0 ]; then
            echo "Validation failed for $cpu_file and $cuda_file"
            exit 1
        fi
    done
done
