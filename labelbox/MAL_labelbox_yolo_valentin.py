import json
import labelbox as lb

import datetime as dt
import os
import sys
import random
import time
from itertools import cycle
from uuid import uuid4
import requests
from pprint import pprint
from multiprocessing.pool import ThreadPool
import os, os.path
import numpy as np
import cv2
from skimage import io
import simplejson as json
import random
from datetime import datetime
import time
import shutil
from matplotlib import pyplot as plt
from pycocotools import mask
import progressbar
from PIL import Image
from google.cloud import storage
import labelbox as lb
from labelbox import Project

from labelbox.schema.bulk_import_request import BulkImportRequest
from labelbox.schema.enums import BulkImportRequestState

import torch


LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3RyMWl3MGsyMTd2MHljdmM4dGwwZ2tpIiwib3JnYW5pemF0aW9uSWQiOiJja3RyMWl3MDEyMTd1MHljdjI0aGE4MjNxIiwiYXBpS2V5SWQiOiJjbDBncnVzczBwMWlqMTAycDF5ZWk1dXI0Iiwic2VjcmV0IjoiZTMxYzQwZjRkZjkyYTkyM2RlZGIxNTk2MjczMDhkNjMiLCJpYXQiOjE2NDY2NjE2OTEsImV4cCI6MjI3NzgxMzY5MX0.FHpoDeRUoyS10VEigBFH74ibxWPWvpcbE4A_HSX5qjo"
#PROJECT_ID = "ckyogza7r9ij010629l9xalt9"
PROJECT_ID = "clfa1qlma08zc07zh3s8d4iox"
DATA_LOCATION = 'seg-data/'
#MODE = 'segmentation-rle'
DATASETS = ['clfa1snsr0qgq07zn6t1190ex', 'clfl8g1we01m607zyc9f2dr4t'] # Add new Dataset ID 
MODE = 'object-detection'
#DATA_LOCATION = 'obj-data/'
#PROJECT_ID='ckw28rrvl64ec0zeehxcn4dic'

DOWNLOAD_IMAGES = True # Download data from labelbox. Set false for re-runs when data already exists locally
VALIDATION_RATIO = 0.2 # Validation data / training data ratio
NUM_CPU_THREADS = 8 # for multiprocess downloads
NUM_SAMPLE_LABELS = 0 # Use 0 to use all of the labeled training data from project. Otherwise specify number of labeled images to use. Use smaller number for faster iteration.
PRELABELING_THRESHOLD = 0.6 # minimum model inference confidence threshold to be uploaded to labelbox
HEADLESS_MODE = False # Set True to skip previewing data or model results

DETECTRON_DATASET_TRAINING_NAME = 'prelabeling-train'
DETECTRON_DATASET_VALIDATION_NAME = 'prelabeling-val'

## get project ontology from labelbox
def get_ontology(project_id):
    response = client.execute(
                """
                query getOntology (
                    $project_id : ID!){ 
                    project (where: { id: $project_id }) { 
                        ontology { 
                            normalized 
                        } 
                    }
                }
                """,
                {"project_id": project_id})
            
    ontology = response['project']['ontology']['normalized']['tools']

    ##Return list of tools and embed category id to be used to map classname during training and inference
    mapped_ontology = []
    thing_classes = []
    
    i=0
    for item in ontology:
#         if item['tool']=='superpixel' or item['tool']=='rectangle':
        item.update({'category': i})
        mapped_ontology.append(item)
        thing_classes.append(item['name'])
        i=i+1         

    return mapped_ontology, thing_classes

## Creates a new export request to get all labels from labelbox. 
def get_labels(project_id):
    should_poll = 1
    while(should_poll == 1):
        response = client.execute(
                    """
                    mutation export(
                    $project_id : ID!    
                    )
                    { 
                        exportLabels(data:{ projectId: $project_id }){ 
                            downloadUrl 
                            createdAt 
                            shouldPoll 
                        }
                    }
                    """,
                    {"project_id": project_id})
        
        if response['exportLabels']['shouldPoll'] == False:
            should_poll = 0
            url = response['exportLabels']['downloadUrl']
            headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}

            r = requests.get(url, headers=headers)
            
            print('Export generated')
            ## writing export to disc for easier debugging
            open('export.json', 'wb').write(r.content)
            return r.content
        else:
            print('Waiting for export generation. Will check back in 10 seconds.')    
            time.sleep(10)

    return response

## Get all previous predictions import (bulk import request). 
def get_current_import_requests():
    response = client.execute(
                    """
                    query get_all_import_requests(
                        $project_id : ID! 
                    ) {
                      bulkImportRequests(where: {projectId: $project_id}) {
                        id
                        name
                      }
                    }
                    """,
                    {"project_id": PROJECT_ID})
    
    return response['bulkImportRequests']

## Delete all current predictions in a project and dataset. We want to delete them and start fresh with predictions from the latest model iteration
def delete_import_request(import_request_id):
    response = client.execute(
                    """
                        mutation delete_import_request(
                            $import_request_id : ID! 
                        ){
                          deleteBulkImportRequest(where: {id: $import_request_id}) {
                            id
                            name
                          }
                        }
                    """,
                    {"import_request_id": import_request_id})
    
    return response

## function to return the difference between two lists. This is used to compute the queued datarows to be used for inference. 
def diff_lists(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 

## Generic data download function
def download_files(filemap):
    path, uri = filemap    
    ## Download data
    if not os.path.exists(path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return path

## Converts binary image mask into COCO RLE format
def rle_encode(mask_image):
    size = list(mask_image.shape)
    pixels = mask_image.flatten()
    
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    
    rle = {'counts': runs.tolist(), 'size': size}
    return rle


def load_set(dir):
    with open(dir+"dataset.json") as json_file:
        dataset_dicts = json.loads(json_file)
    return dataset_dicts

def cv2_imshow(a, **kwargs):
#     a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)

def upload_to_gcs(file_name):
    bucket = storage_client.get_bucket("labelbox-seg")
    blob = bucket.blob("{}.png".format(str(uuid4())))
    blob.upload_from_filename(file_name)
    return blob.generate_signed_url(dt.timedelta(weeks=10))

def mask_to_cloud(img, mask_array, filename):
    num_instances = mask_array.shape[0]
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []
    output = np.zeros_like(img)
    for i in range(num_instances):
        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        output = np.where(mask_array_instance[i] == True, 255, output)
    im = Image.fromarray(output)
    im.save(DATA_LOCATION+'tmp/'+filename+'.png')
    
    cloud_mask = upload_to_gcs(DATA_LOCATION+'tmp/'+filename+'.png')
#     plt.imshow(im)
    return cloud_mask

def convert_yolo_to_coco(yolo_annotations):
    import numpy
    # Initialize category list
    categories = {}
    coco_dataset = {}
    coco_dataset["images"] = []
    coco_dataset["categories"] = []
    coco_dataset["annotations"] = []

    # Loop through YOLO annotations and add them to COCO dataset
    for i, annotations in enumerate(yolo_annotations):
        
        # Add image to COCO dataset
        image_data = {
            "id": i,
            "coco_url": None,
            "date_captured": None
        }
        coco_dataset["images"].append(image_data)

        # Loop through object annotations
        object_annotations = annotations.detach().cpu()
        object_annotations = object_annotations.numpy()
       

        # Get object class and bounding box coordinates
        class_id = int(object_annotations[5])
        x_min = float(object_annotations[0])
        y_min = float(object_annotations[1])
        x_max = float(object_annotations[2])
        y_max = float(object_annotations[3])


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

    return coco_dataset

start_time = time.time()
client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")
# storage_client = storage.Client()
project = client.get_project(PROJECT_ID)

ontology, thing_classes = get_ontology(PROJECT_ID)
print('Available classes: ', thing_classes)
print(ontology)

labels = json.loads(get_labels(PROJECT_ID))

if NUM_SAMPLE_LABELS !=0:
    val_sample = int(VALIDATION_RATIO*NUM_SAMPLE_LABELS)
    val_labels = random.sample(labels, val_sample)
    train_labels = random.sample(labels, NUM_SAMPLE_LABELS)
else:
    split = int(VALIDATION_RATIO*len(labels))
    val_labels = labels[:split]
    train_labels = labels[split:]

if not os.path.exists(DATA_LOCATION+"/inference"):
    os.makedirs(DATA_LOCATION+"/inference")

## Get datarows that needs to be pre-labeled. We are performing a subtraction (all datarows in project - labeled datarows)
datarow_ids_with_labels = []

for label in labels:
    datarow_ids_with_labels.append(label['DataRow ID'])
    
all_datarow_ids = []
all_datarows = []

for dataset_id in DATASETS:
    dataset = client.get_dataset(dataset_id)
    for data_row in dataset.data_rows():
        all_datarow_ids.append(data_row.uid)
        all_datarows.append(data_row)

datarow_ids_queued = diff_lists(all_datarow_ids, datarow_ids_with_labels)

print('Number of datarows to be pre-labeled: ', len(datarow_ids_queued))

## Download queued datarows that needs to be pre-labeled

data_row_queued = []
data_row_queued_urls = []

for datarow in all_datarows:
    for datarow_id in datarow_ids_queued:
        if datarow.uid == datarow_id:
            data_row_queued.append(datarow)
            extension = os.path.splitext(datarow.external_id)[1]
            filename = datarow.uid + extension
            data_row_queued_urls.append((DATA_LOCATION+'inference/' + filename, datarow.row_data))

print('Downloading queued data for inferencing...\n')
filepath_inference = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, data_row_queued_urls)
print('Success...\n')

## Inferencing on queued datarows and create labelbox annotation import file (https://labelbox.com/docs/automation/model-assisted-labeling)

predictions = []
counter = 1

print("Inferencing...\n")
time.sleep(1)
bar = progressbar.ProgressBar(maxval=len(data_row_queued), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

yolomodel = torch.hub.load('/home/GPU/vvial/local_storage/bacteria_tracker/yolov5', 'custom', path='/home/GPU/vvial/local_storage/bacteria_tracker/yolov5/runs/train/yolov5s_500IMGLB/weights/best.pt', source='local') 
#clfl8ixvw01fh07b18t7agl6f
for datarow in data_row_queued:
    extension = os.path.splitext(datarow.external_id)[1]
    filename = "./"+DATA_LOCATION+'inference/' + datarow.uid + extension
    #im = cv2.imread(filename)
    print(filename)


    ##Predict using my model
    res = yolomodel(filename)

    raws_outputs = res.xyxy[0] #xmin ymin xmax ymax confidence class
    outputs = convert_yolo_to_coco(raws_outputs)

    categories = outputs["categories"]
    predicted_boxes = outputs["annotations"]

    if len(categories) != 0:
        for i in range(len(categories)):

            classname = thing_classes[0]

            for item in ontology:
                if classname==item['name']:
                    schema_id = item['featureSchemaId']

        for j in range(len(predicted_boxes)) :
            bbox = predicted_boxes[j]["bbox"]
            bbox_dimensions = {'left': int(bbox[0]), 'top': int(bbox[1]), 'width': int(bbox[2]), 'height': int(bbox[3])}
            predictions.append({"uuid": str(uuid4()),'schemaId': schema_id, 'bbox': bbox_dimensions, 'dataRow': { 'id': datarow.uid }})
    

    print('\predicted '+ str(counter) + ' of ' + str(len(data_row_queued)))
    bar.update(counter)
    counter = counter + 1
    

bar.finish()
time.sleep(1)
print('Total annotations predicted: ', len(predictions))


#Upload predicted annotations to Labelbox project
now = datetime.now() # current date and time
job_name = 'pre-labeling-' + str(now.strftime("%m-%d-%Y-%H-%M-%S"))

#PROJECT_ID = "cl0fj4rbvhec9102p56bjdo09"
#project = client.get_project(PROJECT_ID)

upload_job = project.upload_annotations(
    name=job_name,
    annotations=predictions)

print(upload_job)

upload_job.wait_until_done()

print("State", upload_job.state)

print("--- Finished in %s seconds ---" % (time.time() - start_time))