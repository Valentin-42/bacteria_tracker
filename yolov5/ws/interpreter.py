import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import cv2
from math import sqrt
import os
import sys
import json


def statistics_from_csv(file_path,output_folder_path) :
    # Open the CSV file
    df = pd.read_csv(file_path, compression="zip")

    # Extract the rows
    rows = df.values.tolist()
    
    T = df['Perdu a l\'image'] - df['Apparition a l\'image']
    O = df['Orientations'].tolist()
    
    O_avg = []
    for array in O :
        if not len(ast.literal_eval(array)) > 0 :
            O_avg.append(0)
            continue
        O_avg.append(sum(ast.literal_eval(array))/len(ast.literal_eval(array)))

    # Define the boundaries of the angle categories
    angle_bins = [0, 45, 90, 135]

    O_categories = np.digitize(O_avg, angle_bins)
    for i,angle in enumerate(O_avg) :
        if angle > (90+135)/2 :
            angle = 135
        elif angle > (45+90)/2 :
            angle = 90
        elif angle > (45+0)/2 :
            angle = 40
        else :
            angle = 0
        O_avg[i] = angle

    T_avg = [0,0,0,0]
    T_len  = [0,0,0,0]
    for i,avg_angle in enumerate(O_avg) :
        O_avg[i] = angle_bins[O_categories[i]-1]
        T_avg[O_categories[i]-1] += T[i]
        T_len[O_categories[i]-1] += 1

    for i in range(0,4) :
        T_avg[i] = T_avg[i]/T_len[i]

    print(T_avg)
    # Plot 'Perdu a l'image' vs 'Orientations'
    plt.bar(angle_bins,T_avg,width=10)
    plt.xlabel('O_avg')
    plt.xticks(angle_bins, ['0', '45', '90', '-45'])
    plt.ylabel('Average Tracked Time (in frame)')
    plt.savefig(os.path.join(output_folder_path,'results.png'))


# This function still has a problem somewhere, probably an index error or we are misreading the csv file.
def create_illustration_video(file_path,output_folder_path,metadata_file_path):
    
    with open(metadata_file_path) as f:
        data = json.load(f)

    image_path = data["original_images"]
    duration = data["video_duration"]

    img = cv2.imread(os.path.join(image_path,os.listdir(image_path)[0]))
    height, width, _ = img.shape

    print("Image height:", height)
    print("Image width:", width)
    print("Duration:", duration)

    video = []
    
    # Open the CSV file
    df = pd.read_csv(file_path, compression="zip")

    # Extract the rows
    rows = df.values.tolist()

    print('Reading CSV of lines = '+str(len(rows)))

    appear_frame = []
    lost_frame = []
    orientations = []
    moments=[]
    bb = []
    for array in rows :
        orientations.append(ast.literal_eval(array[1]))
        moments.append(ast.literal_eval(array[5]))
        bb.append(ast.literal_eval(array[4]))
        appear_frame.append(int(array[2]))
        lost_frame.append(int(array[3]))
    print('done')

    print(len(bb))

    for i in range(0,duration) :
        frame = 255*np.ones((height,width,3), np.uint8)
        print("creating img :" +str(i))
        for j in range(0,len(rows)) : # bacteria j

            if len(moments[j])==0:
                continue

            if  i >= appear_frame[j]  and i < lost_frame[j] : #Bacteria j is on the image i  

                index = i-appear_frame[j]

                print("bact :"+str(j))
                print("app frame :"+str(appear_frame[j]))
                print("index: "+str(index))
                print("m size: "+str(len(moments[j])))

                print("m00: "+ str(moments[j][index]["m00"]))

                try :
                    bx =  moments[j][index]["m10"] / moments[j][index]["m00"]
                    by =  moments[j][index]["m01"] / moments[j][index]["m00"]

                    # Central moments 
                    a = moments[j][index]["m20"]/moments[j][index]["m00"] - bx*bx
                    b = 2*(moments[j][index]["m11"]/moments[j][index]["m00"] - bx*by)
                    c = moments[j][index]["m02"]/moments[j][index]["m00"] - by*by

                    #Length
                    minorL = sqrt(8*(a+c-sqrt(b**2+(a-c)**2)))/2
                    majorL = sqrt(8*(a+c+sqrt(b**2+(a-c)**2)))/2
                except :
                    print("cannot cal :" +str(index)+" "+str(moments[j]))


                if(bb[j] != []) :
                    try :
                        center_coordinates = (int(bb[j][index][0]),int(bb[j][index][1]))
                        axesLength = (int(majorL),int(minorL))
                        angle = orientations[j][index]

                        frame = cv2.ellipse(frame, center_coordinates, axesLength, angle, 0, 360, (0,0,255), -1)
                    except :
                        print("cannot bb :" +str(index)+" "+str(len(bb[j])))
        video.append(frame)

    for i,frame in enumerate(video) :
        cv2.imwrite(os.path.join(output_folder_path,"%06d_img.jpg"%i), frame)
        print("saving img :" +str(i))
        

if __name__ == "__main__":

    #base_folder    = '/home/GPU/vvial/local_storage/bacteria_tracker/yolov5/runs/detect/testing/'

    base_folder = sys.argv[1]
    csv_file_path = os.path.join(base_folder,"tracking","data","results.csv")

    output_folder_path = os.path.join(base_folder,"illustration")
    output_image_folder_path = os.path.join(base_folder,"illustration","images")
    data_folder_path = os.path.join(base_folder,"tracking","data")
    metadata_file_path = os.path.join(data_folder_path,"tracking_metadata.json")

    if not os.path.isdir(output_folder_path) :
        os.mkdir(output_folder_path)
    if not os.path.isdir(output_image_folder_path) :
        os.mkdir(output_image_folder_path)

    statistics_from_csv(csv_file_path,data_folder_path)

    # create_illustration_video(csv_file_path,output_image_folder_path,metadata_file_path)
