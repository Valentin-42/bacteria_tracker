#!/bin/bash
echo "Training with yolov5"
python train.py --img 256 --batch 128 --epochs 524288 --data 1bacteria.yaml --weights runs/train/yolov5s_fulldataset5/weights/last.pt --name yolov5s_fulldataset --patience 0