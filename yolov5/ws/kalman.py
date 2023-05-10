import cv2
import numpy
from numpy import dot, asmatrix
import os
import pandas as pd
from math import sqrt
import uuid
import copy
import shutil
import json
import sys

class Bacterie:
    def __init__(self, bb, P, etat):
        self._bb = bb
        self.etat = etat
        self.P = P
        self.counter = 0
        self.orientation = -1

        # For csv file
        self.spawn_frame = -1
        self.lost_frame = -1
        self._center = [[self._bb[0], self._bb[1]]]
        self._boundingboxes = []
        self.orientations = []
        self.moments=[]
        self.ellipticity=-1

    @property
    def bb(self):
        return self._bb
    
    @bb.setter
    def bb(self, value):
        self._bb = value
        self._center.append([self._bb[0], self._bb[1]])
        self._boundingboxes.append(value)

    @property
    def center(self):
        return self._center

    @property
    def boundingboxes(self):
        return self._boundingboxes

    def calculate_moments(self,image,debug=False):
        # Extract the region of interest defined by the bounding box
        x, y, w, h = self.bb
        w_max,h_max,_ = image.shape

        roi = image[max(0,int(y-h/2)):min(h_max,int(y+h/2)), max(0,int(x-w/2)):min(w_max,int(x+w/2))] # Extract ROI
        if not roi.size:
            return
        gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) # To Gray
        mean_val = cv2.mean(gray_img)[0] # Get mean value of pixels
        _, binary_img = cv2.threshold(gray_img,mean_val, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # To binary

        # Calculate the moments of the region of interest
        moments = cv2.moments(binary_img)
        if moments["m00"] == 0 :
            self.orientation = -1
            return

        bx =  moments["m10"] / moments["m00"]
        by =  moments["m01"] / moments["m00"]
        
        # Central moments 
        a = moments["m20"]/moments["m00"] - bx*bx
        b = 2*(moments["m11"]/moments["m00"] - bx*by)
        c = moments["m02"]/moments["m00"] - by*by

        #Length
        minorL = sqrt(8*(a+c-sqrt(b**2+(a-c)**2)))/2
        majorL = sqrt(8*(a+c+sqrt(b**2+(a-c)**2)))/2
        if not majorL == -minorL : 
            ellipticity = (majorL - minorL) / (majorL + minorL)
        else :
            ellipticity =0

        # Calculate the orientation angle of the region of interest
        if moments['mu02'] - moments['mu20'] != 0:
            angle = int(-0.5 * cv2.fastAtan2(2*moments['mu11'], moments['mu20']-moments['mu02'])+180) #+ (a<c)*numpy.pi/2) 
        else:
            angle = 0

        if(debug) :
            print("Estimated angle : "+str(angle))
            unique_filename = str(uuid.uuid4())
            oi = roi.copy()
            oi = cv2.ellipse(oi, (int(bx),int(by)), (int(majorL),int(minorL)),int(-angle+180), 0, 360, (0,0,255), 2)
            cv2.imwrite(os.path.join("test","angle_"+str(round(angle,10))+"_"+unique_filename+".jpg"), oi)

        self.orientations.append(angle)
        self.ellipticity = ellipticity
        self.moments.append(moments)

Llost  = []
data = [] # CSV data


def save_to_csv(X,Llost,filename,video_duration):

    for b in X:
        if(b.lost_frame==-1):
            b.lost_frame = video_duration
        data.append({'Orientations': b.orientations,'Apparition a l\'image': b.spawn_frame, 'Perdu a l\'image': b.lost_frame, 'BB': b.boundingboxes, 'Moments': b.moments})
    
    for b in Llost:
        data.append({'Orientations': b.orientations,'Apparition a l\'image': b.spawn_frame, 'Perdu a l\'image': b.lost_frame, 'BB': b.boundingboxes, 'Moments': b.moments})

    df = pd.DataFrame(data)
    df.to_csv(filename, index=True)


def compute_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
    
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = float(interArea / float(boxAArea + boxBArea - interArea))
        return iou


def Compare(X,Z,img,frame_number) :

    Xb = [] #liste des bacteries apres match
    Xn = []
    eps = 0.5 #Tune : trust factor matching
    cmax = 4
    Zc = Z.copy()
    for zk in Zc : #zk is a bb
        zbk = Bacterie(zk,numpy.zeros((4,4)),'n')
        for xk in X : #xk is a Bacteria
            zbk.calculate_moments(img,False)
            xk.calculate_moments(img,False)
            coef = (zbk.ellipticity + xk.ellipticity)/2
            diff = abs(xk.orientation - zbk.orientation) % 180
            if diff > 90:
                diff = 180 - diff
            diff = (179-diff)/179

            r = numpy.linalg.norm(numpy.array([zbk.bb[0],zbk.bb[1]])- numpy.array([xk.bb[0],xk.bb[1]]))
            k=0.01 #Tune max distance of center of 1 bacteria between 2 frames 
            weight = coef*numpy.exp(-k*(1.1-coef)*r)*(diff)  + (1-coef)*compute_iou(xk.bb, zk)
            if weight > eps :
                xk.etat = 'a'
                if (coef) > (1-coef) : 
                    xk.etat = 'i'
                Z.remove(zk)
                Update(xk,zk)
                Xb.append(xk)
                break
    
    for zk in Z :
        #Create new bacteria
        b = copy.copy(Bacterie(zk,numpy.zeros((4,4)),'n'))
        b.spawn_frame = frame_number
        Xb.append(b)

    for xk in X :
        if xk not in Xb:
            #Lost
            if(xk.counter > cmax) :
                #really lost
                xk.etat = 'l'
                xk.lost_frame = frame_number
                Llost.append(xk)
            else :
                #pas encore perdu
                xk.counter +=1
                Xb.append(xk)
    return Xb


def Predict(X) :
    Q = 10* numpy.identity(4)
    for b in X :
        b.P += Q


def Update(bacteria,Z) : 
    # bacteria = class and Z  = bounding box corresponding to bacteria
    R = 20*numpy.identity(4) # Uncertainty of mesurement
    K = asmatrix(bacteria.P) * asmatrix(numpy.linalg.inv(bacteria.P + R))
    diff = (numpy.array(Z)- numpy.array(bacteria.bb))
    gainb =  numpy.transpose(dot(K, numpy.transpose(diff))).tolist() 
    gainb = [item for sublist in  gainb  for item in sublist] #Flatten
    bacteria.bb = (numpy.array(bacteria.bb) + numpy.array(gainb)).tolist()
    bacteria.P =  dot((numpy.identity(4) - K),bacteria.P)


def main_tracker(path_to_labels,path_to_img,output_folder) :
    
    files = os.listdir(path_to_labels)
    # sorted_files = sorted(files, key=lambda x: int(x[3:-4]))
    sorted_files = sorted(files)
    video_duration = len(files)

    for i,file in enumerate(sorted_files) : #for each frame 
        name,ext = os.path.splitext(file) 
        print("Kalman is processing image : ",name)
        img = cv2.imread(os.path.join(path_to_img,name+".jpg"))
        imgh, imgw = img.shape[:2]
        img_out = img.copy()
        Z = []
        with open(os.path.join(path_to_labels,file), "r") as labelfile:
            for line in labelfile :
                t = line.split("\n")[0].split(" ")
                [x,y,w,h] = [float(t[1]),float(t[2]),float(t[3]),float(t[4])]
                [x,y,w,h] = [int(x*imgw),int(y*imgh),int(w*imgw) ,int(h*imgh)] #Normalize
                Z.append([x,y,w,h])

        if i == 0 :
            X = []
            for bb in Z :
                new_b = Bacterie(bb, numpy.zeros((4,4)),'n')
                new_b.spawn_frame = 0
                X.append(new_b)
        else :
            Predict(X)
            X = Compare(X,Z,img,frame_number=i)

        for bacteria in X :
            bacteria.calculate_moments(img,False)
            [x,y,w,h] =  [bacteria.bb[0],bacteria.bb[1],bacteria.bb[2],bacteria.bb[3]]
            [x,y,w,h] =  [int(x),int(y),int(w),int(h)]
            if bacteria.etat == 'n' :
                color =(0,0,255) #Red
            elif bacteria.etat == 'i' :
                color =(0,255,0) #Green
            else :
                color =(255,0,0) #Blue
            
            cv2.rectangle(img_out, (x-int(w/2), y-int(h/2)), (x+int(w/2), y+int(h/2)), color, 3)
        cv2.imwrite(os.path.join(output_folder,"images",name+".jpg"), img_out)

    save_to_csv(X,Llost,os.path.join(output_folder,"data","results.csv"),video_duration)
    metadata = {'video_duration':video_duration,'original_labels':path_to_labels,'original_images':path_to_img}
    with open(os.path.join(output_folder,"data","tracking_metadata.json"),"w") as f :
        json.dump(metadata,f)

if __name__ == "__main__":

    #Architecture of folders example
    # base_path = '/home/GPU/vvial/local_storage/bacteria_tracker/yolov5/runs/detect/bacteria_raw_yolov5s_500IMGLB/'
    # path_to_img    = '/home/GPU/vvial/home_gtl/bacteria_tracker_ws/raw/bacteria_raw/'
    # output_folder    = '/home/GPU/vvial/home_gtl/bacteria_tracker_ws/experiment/'

    base_path = sys.argv[1]
    path_to_labels = os.path.join(base_path,"labels")
    path_to_img = os.path.join(base_path,"original_images","images")
    output_folder = sys.argv[2]

    if os.path.isdir(output_folder) :
        shutil.rmtree("./test/")
        os.mkdir("./test/")
        if not os.path.isdir(os.path.join(output_folder,"images")):
            os.mkdir(os.path.join(output_folder,"images"))
        if not os.path.isdir(os.path.join(output_folder,"data")):
            os.mkdir(os.path.join(output_folder,"data"))

        print("Results going in : "+str(output_folder))
        
    else :
        print("Output folder not found at :"+str(output_folder))

    main_tracker(path_to_labels,path_to_img,output_folder)
