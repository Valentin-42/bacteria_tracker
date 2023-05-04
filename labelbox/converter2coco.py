import json
import sys
import os

def convert_yolo_to_coco(yolo_path, image_folder, output_path):
    # Initialize COCO dataset
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # Load YOLO annotations and image information
    with open(yolo_path, "r") as f:
        yolo_annotations = f.readlines()
    image_files = os.listdir(image_folder)

    # Initialize category list
    categories = {}

    # Loop through YOLO annotations and add them to COCO dataset
    for i, annotation in enumerate(yolo_annotations):
        # Get image file name
        image_file_name = os.path.splitext(os.path.basename(image_files[i]))[0]

        # Get image size
        image_path = os.path.join(image_folder, image_files[i])
        image_size = os.path.getsize(image_path)

        # Add image to COCO dataset
        image_data = {
            "id": i,
            "file_name": image_file_name,
            "width": 0,
            "height": 0,
            "size": image_size,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None
        }
        coco_dataset["images"].append(image_data)

        # Loop through object annotations
        object_annotations = annotation.strip().split(" ")
        num_objects = int(len(object_annotations) / 5)
        for j in range(num_objects):
            # Get object class and bounding box coordinates
            class_id = int(object_annotations[j * 5])
            x_center = float(object_annotations[j * 5 + 1])
            y_center = float(object_annotations[j * 5 + 2])
            width = float(object_annotations[j * 5 + 3])
            height = float(object_annotations[j * 5 + 4])

            # Convert YOLO coordinates to COCO coordinates
            x_min = int((x_center - (width / 2)) * image_size)
            y_min = int((y_center - (height / 2)) * image_size)
            x_max = int((x_center + (width / 2)) * image_size)
            y_max = int((y_center + (height / 2)) * image_size)

            # Add category to category list if not already present
            if class_id not in categories:
                categories[class_id] = {
                    "id": class_id,
                    "name": "class_" + str(class_id),
                    "supercategory": None
                }
                coco_dataset["categories"].append(categories[class_id])

            # Add object annotation to COCO dataset
            annotation_data = {
                "id": len(coco_dataset["annotations"]),
                "image_id": i,
                "category_id": class_id,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "segmentation": [],
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(annotation_data)

    # Write COCO dataset to file
    with open(output_path, "w") as f:
        json.dump(coco_dataset, f)


yolo_path = sys.argv[1]
img_path = sys.argv[2] 
output = sys.argv[3] 

convert_yolo_to_coco(yolo_path, img_path, output)
