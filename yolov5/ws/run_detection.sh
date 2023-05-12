#!/bin/bash

# set -e -x

echo "Sourcing env '$*'"
set -x
source set_env.sh "$1" "$2" "$3"
set +x
echo "Running detection"

python /mnt/gpu_storage/bacteria_tracker/yolov5/detect.py --source "$images_path" --weights "$model_weights_path" --name "$project_name" --save-txt --hide-labels

rsync -vaP "$images_path"/*_raw.jpg "$original_images_folder"

# Move all images in detect_path_folder to detect_images_folder
mv "${base_path}"/*.jpg "$detect_images_folder"

# Move the detect_labels_folder inside detect_path_folder
mv "$label_path" "$detect_path_folder"
