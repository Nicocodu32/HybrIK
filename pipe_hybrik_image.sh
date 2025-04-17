#!/bin/bash

# This script is used to run the HybrIK pipeline for a given input image and gpu or cpu.
# It sets up the environment, runs the HybrIK pipeline, and then runs the post-processing script.
# Usage: ./pipe_hybrik.sh <input_image> <gpu_or_cpu>
# Example: ./pipe_hybrik.sh image_0 cpu
# Check if the correct number of arguments is provided
# Image must be .png or .jpg
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_image> <gpu_or_cpu>"
    exit 1
fi
# Get the input image and gpu or cpu from the command line arguments
input_image=$1
gpu_or_cpu=$2
# Set the path to the HybrIK repository
repo_path=$(dirname "$(realpath "$0")")
# Set the path to the HybrIK data directory
data_path="$repo_path/examples"
# Set the path to the HybrIK output directory
output_path="$repo_path/data_gen"

# Check if the input image exists
if [ ! -f "examples/$input_image.png" ] && [ ! -f "examples/$input_image.jpg" ]; then
    echo "Error: Input image $input_image does not exist or is not in the right format."
    exit 1
fi

# Check if the output directory exists, if not create it
if [ ! -d "$output_path" ]; then
    mkdir -p "$output_path"
fi

# Run the HybrIK pipeline depending on type of image
if [ ! -f "examples/$input_image.png" ]; then
    python "$repo_path/scripts/smplx_wo_rendering_image.py" --gpu "$gpu_or_cpu" --image-name "examples/${input_image}.jpg" --out-dir "$output_path" --save-pk
else
    python "$repo_path/scripts/smplx_wo_rendering_image.py" --gpu "$gpu_or_cpu" --image-name "examples/${input_image}.png" --out-dir "$output_path" --save-pk
fi

# Check if the HybrIK pipeline ran successfully
if [ $? -ne 0 ]; then
    echo "Error: HybrIK pipeline failed."
    exit 1
fi

# Check if the output post-processing directory exists, if not create it
if [ ! -d "$repo_path/../rt-cosmik/src/rtcosmik/smpl/hybrik/smplx/res_${input_image}_x" ]; then
    mkdir -p "$repo_path/../rt-cosmik/src/rtcosmik/smpl/hybrik/smplx/res_${input_image}_x"
fi

# Run the HybrIK post-processing scripts
python "$repo_path/../rt-cosmik/src/rtcosmik/smpl/smpl_data_parsing/smplx_hybrik_to_markers.py" --data-path "$output_path/res_${input_image}_x.pk"
python "$repo_path/../rt-cosmik/src/rtcosmik/smpl/smpl_data_parsing/smpl_mks_to_lstm_mks.py" --data-path "$repo_path/../rt-cosmik/src/rtcosmik/smpl/hybrik/res_${input_image}_x_mks_2.csv"

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
