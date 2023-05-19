import labelbox as lb
import labelbox.types as lb_types
import uuid
import numpy as np
import json
import os 
from PIL import Image
from urllib.request import urlopen

def export(output_folder) :

    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)

    API_KEY = ""
    client = lb.Client(API_KEY)

    project_id = "clfa1qlma08zc07zh3s8d4iox"
    project = client.get_project(project_id)

    labels = project.export_labels(download=True)
    save(labels)
    for label in labels:
        img_url = label['Labeled Data']
        image_filename = label['External ID']

        image_fullpath=os.path.join(output_folder,image_filename)
        if os.path.exists(image_fullpath):
            img = Image.open(image_fullpath)
        else:
            img = Image.open(urlopen(img_url))
        bbox = img.getbbox()
        if bbox is None:
            print("Skipping '%s'" % image_filename)
            continue
        if not os.path.exists(image_fullpath):
            img.save(image_fullpath)
        split_labels_to_txt_files(label, bbox, output_folder)


def save(labels) :
    with open('labelbox_export.json', 'w') as f:
        json.dump(labels, f)
    print("done")

def split_labels_to_txt_files(data, ibox, out_folder) : 
    
    # Extract the image filename from the 'External ID' key
    image_filename = data['External ID']

    # Remove the file extension from the filename
    image_filename_without_ext = os.path.splitext(image_filename)[0]

    # Create a new file with the same name as the image file
    output_filename = f"{image_filename_without_ext}.txt"
    _,_,iw,ih = ibox
    iw=float(iw)
    ih=float(ih)

    # Open the output file in write mode
    with open(os.path.join(out_folder,output_filename), 'w') as output_file:
        
        # Iterate over the objects in the label data
        for obj in data['Label']['objects']:
            
            # Extract the bounding box coordinates from the 'bbox' key
            bbox = obj['bbox']
            x, y, w, h = (bbox['left']+bbox['width']/2.)/iw, (bbox['top']+bbox['height']/2.)/ih, bbox['width']/iw, bbox['height']/ih
            
            # Write the bounding box coordinates to the output file
            output_file.write(f"0 {x} {y} {w} {h}\n")

    print("done "+str(image_filename))


if __name__ == "__main__":
    output_folder  = "./test/"
    export(output_folder)
