#!/bin/bash

set -x

BASE="$HOME/src/bacteria_tracker"

for d1 in /mnt/bacteria/dataset/5*muL-min-1; do 
    bn1=`basename "$d1"`
    for d2 in "$d1"/*19; do
        bn2=`basename "$d2"`
        bn="${bn1}-${bn2}"
        rm -rf "$BASE"/yolov5/runs/detect/"$bn"
        ./run_detection.sh "${d2}/raw" "$bn" "$BASE"/yolov5/runs/detect
        ./run_kalman.sh "${d2}/raw" "$bn" "$BASE"/yolov5/runs/detect
        ./run_finalize.sh "${d2}/raw" "$bn" "$BASE"/yolov5/runs/detect
    done
done

