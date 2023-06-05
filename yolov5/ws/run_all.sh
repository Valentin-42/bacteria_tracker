#!/bin/bash

set -e

for d in /mnt/bacteria/dataset/*; do 
    bn=`basename "$d"`_cp
    rm -rf /mnt/gpu_storage/bacteria_tracker/yolov5/runs/detect/"$bn"
    ./run_detection.sh "${d}" "$bn" /mnt/gpu_storage/bacteria_tracker/yolov5/runs/detect
    # ./run_kalman.sh "${d}/raw" "$bn" /mnt/gpu_storage/bacteria_tracker/yolov5/runs/detect
    # ./run_finalize.sh "${d}/raw" "$bn" /mnt/gpu_storage/bacteria_tracker/yolov5/runs/detect
done

