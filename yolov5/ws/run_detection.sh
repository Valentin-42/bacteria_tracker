#!/bin/bash

# set -e -x

echo "Sourcing env '$*'"
set -x
source set_env.sh "$1" "$2" "$3" no
set +x
echo "Running detection on '${images_path}'"
echo "Base path ${base_path}"

# set -e

python3 /mnt/gpu_storage/bacteria_tracker/yolov5/detect.py --source "$images_path" --weights "$model_weights_path" --name "$project_name"\
    --project "$result_folder" --data "$model_data_path" --save-txt --hide-labels --max-det 2000 --device 1

source set_env.sh "$1" "$2" "$3" yes

rsync -vaP "$images_path"/*.jpg "$original_images_folder"

# Move all images in detect_path_folder to detect_images_folder
mv "${base_path}"/*.jpg "$detect_images_folder"

# Move the detect_labels_folder inside detect_path_folder
mv "${base_path}/labels" "$detect_path_folder"
