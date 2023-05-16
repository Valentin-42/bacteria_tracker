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
import scipy

BacteriaCounter=0

def GetNewBacteriaId():
    global BacteriaCounter
    val=BacteriaCounter
    BacteriaCounter+=1
    return val

class Bacteria:
    def __init__(self, bb, P, etat,bid):
        global BacteriaCounter
        self.id = bid
        self._bb = bb
        self.etat = etat
        self.P = P
        self.counter = 0
        self.orientation = -1
        self.roi = None
        self.groi = None

        # For csv file
        self.last_seen = -1
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

    def image_has_changed(self,image, frame=0):
        if self.groi is None:
            return True
        x, y, w, h = self.bb
        h_max,w_max,_ = image.shape

        roi = image[max(0,int(y-h/2)):min(h_max,int(y+h/2)), max(0,int(x-w/2)):min(w_max,int(x+w/2))] # Extract ROI
        if not self.roi.size:
            return True
        groi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) # To Gray

        droi = (groi.astype(float)-self.groi.astype(float)).flatten()
        diff2 = numpy.linalg.norm(droi,2) / droi.shape[0]
        diff1 = numpy.linalg.norm(droi,1) / droi.shape[0]
        diffi = numpy.linalg.norm(droi,numpy.inf) 
        std = numpy.std(self.groi)
        # print("Image diff: %f %f %f std %f" % (diff2,diff1,diffi,std))
        # cv2.imwrite("oldroi_%04d_%04d.png" % (self.id,frame),self.groi)
        # cv2.imwrite("newroi_%04d_%04d.png" % (self.id,frame),groi)
        return diffi > 2*std


    def acquire_roi(self,image):
        # if self.id is not None:
        #     print("Bacteria %d: updating roi" % self.id)
        # Extract the region of interest defined by the bounding box
        x, y, w, h = self.bb
        h_max,w_max,_ = image.shape

        self.roi = image[max(0,int(y-h/2)):min(h_max,int(y+h/2)), max(0,int(x-w/2)):min(w_max,int(x+w/2))].copy() # Extract ROI
        if not self.roi.size:
            self.roi = None
            self.groi = None
            return
        self.groi = cv2.cvtColor(self.roi, cv2.COLOR_RGB2GRAY) # To Gray

    def calculate_moments(self,image,debug=False):
        self.acquire_roi(image)
        mean_val = cv2.mean(self.groi)[0] # Get mean value of pixels
        _, binary_img = cv2.threshold(self.groi,mean_val, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # To binary

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
    df.to_csv(filename, index=True, compression="zip")


def compute_dcenter(boxA, boxB):
    cA=[boxA[0],boxA[1]]
    cB=[boxB[0],boxB[1]]
    return numpy.hypot(cA[0]-cB[0],cA[1]-cB[1])

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

    Xb = [] #liste des Bacterias apres match
    Xn = []
    eps = 0.1 #Tune : trust factor matching
    cmax = 2
    cmax2 = 6
    matched=[0]*len(Z)
    Cost=1000*numpy.ones((len(X),len(Z)))

    for iz,zk in enumerate(Z) : #zk is a bb
        zbk = Bacteria(zk,max(zk[2],zk[3])/5 * numpy.identity(4),'n',None)
        zbk.calculate_moments(img,False)
        for ix,xk in enumerate(X) : #xk is a Bacteria
            coef = (zbk.ellipticity + xk.ellipticity)/2
            do = abs(numpy.fmod(xk.orientation - zbk.orientation+270,180)-90)
            diff = (179.-do)/179.

            r = compute_dcenter(xk.bb,zk)
            k=0.01 #Tune max distance of center of 1 bacteria between 2 frames 
            iou = compute_iou(xk.bb, zk)
            if iou < 1e-6:
                continue
            weight = coef*numpy.exp(-k*(1.1-coef)*r)*(diff)  + (1-coef)*iou
            # print("Z %d - X %d: coef %f diff %f %f r %f iou %f w %f" % (iz,xk.id,coef,do,diff,r,iou,weight))
            if weight > eps :
                Cost[ix,iz]=1 - weight

    # print(Cost)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(Cost)
    # print([(r,c) for (r,c) in zip(row_ind,col_ind)])
    for (r,c) in zip(row_ind,col_ind):
        if Cost[r,c] < 1:
            xk = X[r]
            zk = Z[c]
            print("Matching obs %d with bacteria %d" % (c,xk.id))
            xk.etat = 'a'
            if (coef) > (1-coef) : 
                xk.etat = 'i'
            matched[c]=1
            Update(xk,zk)
            xk.calculate_moments(img,False)
            xk.counter = 0
            xk.last_seen = frame_number
            Xb.append(xk)
    
    for iz,(im,zk) in enumerate(zip(matched,Z)) :
        if im == 1:
            continue
        print("Obs %d is a new bacteria" % iz)
        #Create new bacteria
        b = Bacteria(zk,max(zk[2],zk[3])/5 * numpy.identity(4),'n',GetNewBacteriaId())
        b.calculate_moments(img,False)
        b.spawn_frame = frame_number
        b.last_seen = frame_number
        Xb.append(b)

    for xk in X :
        if xk not in Xb:
            xk.counter += 1
            if not xk.image_has_changed(img,frame_number):
                xk.last_seen = frame_number
                if(xk.counter > cmax2) :
                    print("Missing for too long, bacteria %d is lost" % xk.id)
                    #really lost
                    xk.etat = 'l'
                    xk.lost_frame = frame_number
                    Llost.append(xk)
                else:
                    print("Bacteria %d is missing (c=%d) but image did not change" % (xk.id,xk.counter))
                    Xb.append(xk)
                    xk.etat = 'u'
                    xk.calculate_moments(img,False)
            #Lost
            else :
                #pas encore perdu
                if(xk.counter > cmax) :
                    print("Bacteria %d is lost" % xk.id)
                    #really lost
                    xk.etat = 'l'
                    xk.lost_frame = frame_number
                    Llost.append(xk)
                else:
                    print("Bacteria %d is missing (c=%d)" % (xk.id,xk.counter))
                    Xb.append(xk)
                    xk.etat = 'm'
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
    
    #files = os.listdir(path_to_labels)
    files = os.listdir(path_to_img)
    # sorted_files = sorted(files, key=lambda x: int(x[3:-4]))
    sorted_files = sorted(files)
    video_duration = len(files)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
      
    # fontScale
    fontScale = 1
       
    # Line thickness of 2 px
    thickness = 2
       

    X = []
    for i,file in enumerate(sorted_files) : #for each frame 
        # if i > 50:
        #     break
        name,ext = os.path.splitext(file) 
        print("Kalman is processing image : ",name)
        img = cv2.imread(os.path.join(path_to_img,name+".jpg"))
        imgh, imgw = img.shape[:2]
        img_out = img.copy()
        cv2.normalize(img, img_out, 255.0, 0.0, cv2.NORM_MINMAX);
        Z = []
        if os.path.exists(os.path.join(path_to_labels,name+".txt")):
            with open(os.path.join(path_to_labels,name+".txt"), "r") as labelfile:
                for line in labelfile :
                    t = line.strip().split(" ")
                    [x,y,w,h] = [float(t[1]),float(t[2]),float(t[3]),float(t[4])]
                    [x,y,w,h] = [int(x*imgw),int(y*imgh),int(w*imgw) ,int(h*imgh)] #Normalize
                    Z.append([x,y,w,h])

        print("Observations:")
        for iz,bb in enumerate(Z):
            print("%d: %s"%(iz,str(bb)))

        print("Current State:")
        for ix,b in enumerate(X):
            print("%d id %d: %s %s" % (ix,b.id,str([int(round(x)) for x in b.bb]),b.etat))


        if i == 0 :
            for iz,bb in enumerate(Z) :
                print("Init: Obs %d is a new bacteria" % iz)
                new_b = Bacteria(bb, max(bb[2],bb[3])/5 * numpy.identity(4),'n',GetNewBacteriaId())
                new_b.calculate_moments(img,False)
                new_b.spawn_frame = 0
                X.append(new_b)
        else :
            Predict(X)
            X = Compare(X,Z,img,frame_number=i)

        for bacteria in X :
            [x,y,w,h] =  [bacteria.bb[0],bacteria.bb[1],bacteria.bb[2],bacteria.bb[3]]
            [x,y,w,h] =  [int(x),int(y),int(w),int(h)]
            if bacteria.etat == 'n' :
                color =(0,0,255) #Red
            elif bacteria.etat == 'i' :
                color =(0,255,0) #Green
            elif bacteria.etat == 'u' :
                color =(0,128,192) #orange ? 
            elif bacteria.etat == 'm' :
                color =(255,0,255) #purple
            elif bacteria.etat == 'l' :
                color =(128,128,128) #gray
            elif bacteria.etat == 'a' :
                color =(255,255,0) #gray
            else :
                color =(255,255,255) # Should not be here
            
            cv2.rectangle(img_out, (x-int(w/2), y-int(h/2)), (x+int(w/2), y+int(h/2)), color, 3)
            cv2.putText(img_out, str(bacteria.id), (x-int(w/2), y-int(h/2)-6), font, 
                               fontScale, (255,255,255), thickness, cv2.LINE_AA)

        cv2.imwrite(os.path.join(output_folder,"images",name+".jpg"), img_out)

    print("Saving CSV and JSON to disk")
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
    path_to_labels = os.path.join(base_path,"detect","labels")
    path_to_img = os.path.join(base_path,"original_images","images")
    output_folder = sys.argv[2]

    if os.path.isdir(output_folder) :
        try:
            shutil.rmtree("./test/")
        except:
            pass
        os.mkdir("./test/")
        if not os.path.isdir(os.path.join(output_folder,"images")):
            os.mkdir(os.path.join(output_folder,"images"))
        if not os.path.isdir(os.path.join(output_folder,"data")):
            os.mkdir(os.path.join(output_folder,"data"))

        print("Results going in : "+str(output_folder))
        
    else :
        print("Output folder not found at :"+str(output_folder))

    main_tracker(path_to_labels,path_to_img,output_folder)
