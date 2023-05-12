#!/bin/bash

# set -e -x

echo "Sourcing env '$*'"
set -x
source set_env.sh "$1" "$2" "$3"
set +x
echo "Running kalman"

# Tracking images from detections saved
python kalman.py "$base_path" "$tracking_path_folder"
