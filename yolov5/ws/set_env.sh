#!/bin/bash

# echo "$*"

if test "x$1" = "x"
then
    images_path="/mnt/bacteria/Manip2-debut/raw"
else
    images_path="$1"
fi

if test "x$2" = "x"
then
    project_name=`basename "$image_path"`
else
    project_name="$2"
fi

if test "x$3" = "x"
then
    result_folder="/mnt/gpu_storage/bacteria_tracker/yolov5/runs/detect"
else
    result_folder="$3"
fi

create=yes
if test "x$4" = "x"
then
    create=yes
else
    create=$4
fi


# Variables
#images_path="/mnt/bacteria/T1J (9h16-9h33)/raw"
model_weights_path="../runs/train/yolov5s_split_annotated_20230501/weights/best.pt"
model_data_path="../data/2bacteria.yaml"
#project_name="Manip2-debut_cp"

# Create folder architecture
base_path="${result_folder}/${project_name}"

original_path_folder="${base_path}/original_images"
original_images_folder="${original_path_folder}/images"

tracking_path_folder="${base_path}/tracking"
tracking_image_path_folder="${tracking_path_folder}/images"

illustration_path_folder="${base_path}/illustration/"
illustration_image_path_folder="${illustration_path_folder}/images/"

detect_path_folder="${base_path}/detect/"
detect_images_folder="${detect_path_folder}/images/"
label_path="${detect_path_folder}/labels"


# Create workspace
if test "$create" = "yes"
then
    mkdir -p "$original_path_folder"
    mkdir -p "$original_images_folder"

    mkdir -p "$tracking_path_folder"
    mkdir -p "$tracking_image_path_folder"

    mkdir -p "$detect_path_folder"
    mkdir -p "$detect_images_folder"
fi

