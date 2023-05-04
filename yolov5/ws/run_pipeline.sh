#!/bin/bash
echo "Pipeline : Detect -> Kalman -> Interpreter -> Organisation"

# Variables
images_path="/home/GPU/vvial/home_gtl/bacteria_tracker_ws/Videos/T1J(9h16-9h33)_raw/"
model_weights_path="~/local_storage/bacteria_tracker/yolov5/runs/train/yolov5s_fulldataset5/weights/best.pt"
project_name="T1J(9h16-9h33)_raw"

# Inference on images
python ~/local_storage/bacteria_tracker/yolov5/detect.py --source $images_path --weights $model_weights_path --name $project_name --save-txt --hide-labels

# Create folder architecture
base_path="~/local_storage/bacteria_tracker/yolov5/runs/detect/${project_name}/"
label_path="${base_path}labels/"

original_path_folder="${base_path}original_images/"
original_images_folder="${original_path_folder}images/"

tracking_path_folder="${base_path}tracking/"
tracking_image_path_folder="${tracking_path_folder}images/"



# Create workspace
mkdir -p "$original_path_folder"
mkdir -p "$original_images_folder"

mkdir -p "$tracking_path_folder"
mkdir -p "$tracking_image_path_folder"


# Copy all images from the images_path to the destination folder
echo "Setup folder architecture"
cp $images_path* $original_images_folder

# Tracking images from detections saved
python kalman.py $base_path $tracking_path_folder

# Plot
python interpreter.py $base_path

illustration_path_folder="${base_path}illustration/"
illustration_image_path_folder="${illustration_path_folder}images/"

# Reorganisation and export

detect_path_folder="${base_path}detect/"
detect_images_folder="${detect_path_folder}images/"

echo "Reorganisation..."

mkdir -p "$detect_path_folder"
mkdir -p "$detect_images_folder"

# Move all images in detect_path_folder to detect_images_folder
mv ${base_path}*.jpg $detect_images_folder

# Move the detect_labels_folder inside detect_path_folder
mv $label_path $detect_path_folder

#Create Videos
python images_to_video.py $original_images_folder $original_path_folder
python images_to_video.py $tracking_image_path_folder $tracking_path_folder
python images_to_video.py $detect_images_folder $detect_path_folder
python images_to_video.py $illustration_image_path_folder $illustration_path_folder

echo "Done !"
