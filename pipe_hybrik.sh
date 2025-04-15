#!/bin/bash

# This script is used to run the HybrIK pipeline for a given input video and output directory and gpu or not.
# It sets up the environment, runs the HybrIK pipeline, and then runs the post-processing script.
# Usage: ./pipe_hybrik.sh <input_video> <output_directory> <gpu_or_not>
# Example: ./pipe_hybrik.sh input.mp4 output_dir
# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_video> <output_directory> <gpu_or_not>"
    exit 1
fi
# Get the input video and output directory from the command line arguments
input_video=$1
output_dir=$2
gpu_or_not=$3
# Set the path to the HybrIK repository
repo_path=$(dirname "$(realpath "$0")")
# Set the path to the HybrIK data directory
data_path="$repo_path/examples"
# Set the path to the HybrIK output directory
output_path="$repo_path/data_gen"

# Check if the input video exists
if [ ! -f "examples/$input_video.mp4" ]; then
    echo "Error: Input video $input_video does not exist."
    exit 1
fi
# Check if the output directory exists, if not create it
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# Run the HybrIK pipeline
python "$repo_path/scripts/smplx_wo_rendering.py" --gpu "$gpu_or_not" --video-name "examples/${input_video}.mp4" --out-dir "$output_path" --save-pk

# Check if the HybrIK pipeline ran successfully
if [ $? -ne 0 ]; then
    echo "Error: HybrIK pipeline failed."
    exit 1
fi

# Check if the output post-processing directory exists, if not create it
if [ ! -d "$repo_path/../rt-cosmik/src/rtcosmik/smpl/hybrik/smplx/res_${input_video}_x" ]; then
    mkdir -p "$repo_path/../rt-cosmik/src/rtcosmik/smpl/hybrik/smplx/res_${input_video}_x"
fi

# Run the HybrIK post-processing scripts
python "$repo_path/../rt-cosmik/src/rtcosmik/smpl/smpl_data_parsing/smplx_hybrik_to_markers.py" --data-path "$output_path/res_${input_video}_x.pk"
python "$repo_path/../rt-cosmik/src/rtcosmik/smpl/smpl_data_parsing/smpl_mks_to_lstm_mks.py" --data-path "$repo_path/../rt-cosmik/src/rtcosmik/smpl/hybrik/res_${input_video}_x_mks_2.csv"

# Check if the HybrIK post-processing script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: HybrIK post-processing script failed."
    exit 1
fi

# Print a success message
echo "HybrIK pipeline and post-processing completed successfully."
# Exit the script
exit 0
# End of script
