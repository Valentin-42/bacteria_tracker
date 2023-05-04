import os
import sys 
import json

YOLO_ANNOTATIONS_DIR = sys.argv[1]
COCO_ANNOTATIONS_FILE = sys.argv[2] 

CATEGORIES = [
    {"id": 0, "name": "bacteria"},
]

coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": CATEGORIES,
}

for idx, file_name in enumerate(sorted(os.listdir(YOLO_ANNOTATIONS_DIR))):
    if file_name.endswith(".txt"):
        image_name = file_name.split(".")[0]
        image_id = idx 
        with open(os.path.join(YOLO_ANNOTATIONS_DIR, file_name), "r") as f:
            yolo_annotations = f.readlines()

        for yolo_annotation in yolo_annotations:
            class_id, x_center, y_center, width, height = map(float, yolo_annotation.split())

            x_min = (x_center - (width / 2)) * 224
            y_min = (y_center - (height / 2)) * 224
            w = width * 224
            h = height * 224

            annotation = {
                "id": len(coco_annotations["annotations"]),
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            coco_annotations["annotations"].append(annotation)

        image = {
            "id": image_id,
            "width": 224,
            "height": 224,
            "file_name": f"{image_name}.jpg",
        }
        coco_annotations["images"].append(image)

with open(COCO_ANNOTATIONS_FILE, "w") as f:
    json.dump(coco_annotations, f)

