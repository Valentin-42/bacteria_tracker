#!/bin/bash

echo "Sourcing env '$*'"
set -x
source set_env.sh "$1" "$2" "$3"
set +x
echo "Running images_to_video"

# Reorganisation and export

echo "Reorganisation..."


#Create Videos
python images_to_video.py "$original_images_folder" "$original_path_folder"
python images_to_video.py "$tracking_image_path_folder" "$tracking_path_folder"
python images_to_video.py "$detect_images_folder" "$detect_path_folder"
# python images_to_video.py "$illustration_image_path_folder" "$illustration_path_folder"

echo "Done !"
