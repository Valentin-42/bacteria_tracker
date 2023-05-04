import json
import sys 
import os

def converter(input_txt_file_path, output_json_file_path) :

    with open(input_txt_file_path, 'r') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        data = line.strip().split()
        label = int(data[0])
        left, top, width, height = map(float, data[1:])
        right = left + width
        bottom = top + height
        annotations.append({
            'label': label,
            'bbox': [left, top, right, bottom]
        })

    with open(output_json_file_path, 'w') as f:
        for annotation in annotations:
            json.dump(annotation, f)
            f.write('\n')


if __name__ == "__main__":
    base_path = sys.argv[1] 
    path_to_labels = base_path+'predictions/'

    output_folder = 'json/'

    files = os.listdir(path_to_labels)
    sorted_files = sorted(files, key=lambda x: int(x[3:-4]))

    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)
        print("Ouput directory created !")

    for i,file in enumerate(sorted_files) :
        print(file)
        name,ext = os.path.splitext(file) 
        converter(path_to_labels+file,output_folder+name+'.json')
        
