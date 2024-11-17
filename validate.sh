#!/bin/bash

output_folder="output"

# check if the output folder exists
if [ ! -d $output_folder ]; then
    echo "Output folder does not exist"
    exit 1
fi

# check if the output folder is empty
if [ ! "$(ls -A $output_folder)" ]; then
    echo "Output folder is empty"
    exit 1
fi

# check if the output folder contains the cpu and cuda subfolders
if [ ! -d "$output_folder/cpu" ] || [ ! -d "$output_folder/cuda" ]; then
    echo "Output folder does not contain the cpu and cuda subfolders"
    exit 1
fi

# check if the cpu and cuda subfolders are empty
if [ ! "$(ls -A $output_folder/cpu)" ] || [ ! "$(ls -A $output_folder/cuda)" ]; then
    echo "CPU or CUDA subfolder is empty"
    exit 1
fi

# loop through the output/cpu and output/cuda folder together
for cpu_folder in $output_folder/cpu/*; do
    folder_name=$(basename $cpu_folder)
    cuda_folder="${cpu_folder/cpu/cuda}"
    if [ ! -d "$cuda_folder" ]; then
        echo "Missing corresponding CUDA folder for $cpu_folder"
        exit 1
    fi

    # check if the final output files exist in both cpu and cuda folders
    if [ ! -f "$cpu_folder/final" ] || [ ! -f "$cuda_folder/final" ]; then
        echo "Missing final output file in $cpu_folder or $cuda_folder"
        exit 1
    fi

    echo "========== Validating $folder_name =========="

    # loop through the final output files in cpu and cuda folders with .out extension
    for cpu_file in $cpu_folder/*.out; do
        cuda_file="${cpu_file/cpu/cuda}"
        if [ ! -f "$cuda_file" ]; then
            echo "Missing corresponding CUDA file for $cpu_file"
            exit 1
        fi
        
        # run the validation prog, if not exit with error code
        ./validator $cpu_file $cuda_file
        if [ $? -ne 0 ]; then
            echo "Validation failed for $cpu_file and $cuda_file"
            exit 1
        fi
    done
done
