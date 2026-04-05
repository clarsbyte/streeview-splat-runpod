#!/bin/bash
#
# train_speedy_splat.sh
#
# Author: Nandan Manjunatha nannigalaxy@gmail.com
# License: MIT
# Description:
#   This script runs the Speedy Splat training process using images from the
#   SfM pipeline and saves the trained model to a specified directory.
#
# Usage:
#   ./train_speedy_splat.sh <sfm_output_dir> <gsplat_output_dir_path>
#

# Check if enough arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sfm_input_dir> <gsplat_output_dir_path>"
    exit 1
fi

# Read command-line arguments
SFM_INPUT_DIR="$1"  # Directory containing undistorted images from SfM
GSPLAT_OUTPUT_DIR="$2"  # Directory to save the output model

# Create necessary directories if they don't exist
mkdir -p "$GSPLAT_OUTPUT_DIR"

echo "Running Speedy Splat training..."

python3 speedy-splat/train.py \
    --source_path "$SFM_INPUT_DIR" \
    --model_path "$GSPLAT_OUTPUT_DIR" \
    --resolution 1024 \
    --iterations 30000 \
    --position_lr_init 0.00016 \
    --position_lr_final 0.0000016 \
    --feature_lr 0.0025 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    --percent_dense 0.01 \
    --lambda_dssim 0.2 \
    --densification_interval 100 \
    --densify_until_iter 15000 \
    --opacity_reset_interval 3000 \
    --densify_grad_threshold 0.0002 \
    --save_iterations 7000 30000
