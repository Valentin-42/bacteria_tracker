#!/bin/bash
echo "Inference over folder"
python detect.py --source /home/GPU/vvial/local_storage/bacteria_tracker/datasets/4Videos/images/train/ --weights runs/train/yolov5s_500IMGLB/weights/best.pt --name images_yolov5s_500IMGLB --save-txt --hide-labels