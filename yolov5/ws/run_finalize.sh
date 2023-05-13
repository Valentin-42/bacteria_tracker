#!/bin/bash

echo "Sourcing env '$*'"
set -x
source set_env.sh "$1" "$2" "$3"
set +x
echo "Running images_to_video"

# Reorganisation and export

echo "Reorganisation..."


#Create Videos
if which mencoder
then
    #using mjpeg. Handbrake has better parameter to convert to mp4
    mencoder "mf://${original_images_folder}/*.jpg" -mf fps=4 -nosound -ovc lavc -lavcopts vcodec=mjpeg -o "${original_path_folder}/${project_name}.avi" &
    mencoder "mf://${tracking_image_path_folder}/*.jpg" -mf fps=4 -nosound -ovc lavc -lavcopts vcodec=mjpeg -o "${tracking_path_folder}/${project_name}.avi" &
    mencoder "mf://${detect_images_folder}/*.jpg" -mf fps=4 -nosound -ovc lavc -lavcopts vcodec=mjpeg -o "${detect_path_folder}/${project_name}.avi" &
    # wait for completion
    FAIL=0
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
else
    python images_to_video.py "$original_images_folder" "$original_path_folder"
    python images_to_video.py "$tracking_image_path_folder" "$tracking_path_folder"
    python images_to_video.py "$detect_images_folder" "$detect_path_folder"
    # python images_to_video.py "$illustration_image_path_folder" "$illustration_path_folder"
fi

echo "Done !"
