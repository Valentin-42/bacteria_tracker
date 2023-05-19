#!/bin/bash

# set -e -x

echo "Pipeline : Detect -> Kalman -> Interpreter -> Organisation"

# Variables
#images_path="/mnt/bacteria/T1J (9h16-9h33)/raw"
#images_path="/mnt/bacteria/Manip2-debut/raw"
images_path="/mnt/bacteria/Manip2-1h30-adhesion/raw" 
model_weights_path="/mnt/gpu_storage/bacteria_tracker/yolov5/runs/train/yolov5s_fulldataset5/weights/best.pt"
project_name="Manip2-1h30-adhesion_cp"

# Create folder architecture
base_path="/mnt/gpu_storage/bacteria_tracker/yolov5/runs/detect/${project_name}"
label_path="${base_path}/labels"

original_path_folder="${base_path}/original_images"
original_images_folder="${original_path_folder}/images"

tracking_path_folder="${base_path}/tracking"
tracking_image_path_folder="${tracking_path_folder}/images"

illustration_path_folder="${base_path}/illustration/"
illustration_image_path_folder="${illustration_path_folder}/images/"

detect_path_folder="${base_path}/detect/"
detect_images_folder="${detect_path_folder}/images/"

if test -d "$label_path"
then
    echo "Skipping detect stage because \'$label_path\" exists"
else
    # Inference on images
    python /mnt/gpu_storage/bacteria_tracker/yolov5/detect.py --source "$images_path" --weights "$model_weights_path" --name "$project_name" --save-txt --hide-labels
fi

exit 0


# Create workspace
mkdir -p "$original_path_folder"
mkdir -p "$original_images_folder"

mkdir -p "$tracking_path_folder"
mkdir -p "$tracking_image_path_folder"

rsync -vaP "$images_path"/*_raw.jpg "$original_images_folder"

# Copy all images from the images_path to the destination folder
echo "Setup folder architecture"
if test -r "$tracking_path_folder"/data/tracking_metadata.json
then
    echo "Skipping kalman stage because \'$tracking_path_folder/data/tracking_metadata.json\" exists"
else
    # Tracking images from detections saved
    python kalman.py "$base_path" "$tracking_path_folder"
fi

# Plot
if test -d "$illustration_path_folder"
then
    echo "Skipping interpreter stage because \'$illustration_path_folder\" exists"
else
    python interpreter.py "$base_path"
fi


# Reorganisation and export

echo "Reorganisation..."

mkdir -p "$detect_path_folder"
mkdir -p "$detect_images_folder"

# Move all images in detect_path_folder to detect_images_folder
mv "${base_path}"/*.jpg "$detect_images_folder"

# Move the detect_labels_folder inside detect_path_folder
mv "$label_path" "$detect_path_folder"

#Create Videos
python images_to_video.py "$original_images_folder" "$original_path_folder"
python images_to_video.py "$tracking_image_path_folder" "$tracking_path_folder"
python images_to_video.py "$detect_images_folder" "$detect_path_folder"
# python images_to_video.py "$illustration_image_path_folder" "$illustration_path_folder"

echo "Done !"
