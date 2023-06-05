#!/bin/bash
echo "Training with yolov5"
python train.py --img 256 --batch 128 --epochs 524288 --data 2bacteria.yaml --weights runs/train/yolov5s_split_annotated_20230501/weights/best.pt --name yolov5s_split_annotated_20230501_cluster --patience 0 --device 1
