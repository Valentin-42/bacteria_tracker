import cv2
import numpy
from numpy import dot, asmatrix
import os

class Bacterie : 
    def __init__(self,bb,P,etat):
        self.bb = bb
        self.etat = etat
        self.P = P
        self.counter = 0

Llost   = []

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
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        return iou

def Compare(X,Z) :

    Xb = [] #liste des bacterie apres match
    eps = 0.5
    cmax = 2
    for zk in Z : #zk is a bb
        for xk in X : #xk is a Bacteria
            if compute_iou(xk.bb, zk) < eps :
                xk.etat = 'a'      
                Z.remove(zk)
                X.remove(xk)
                Update(xk,zk)
                Xb.append(xk)
                break
    
    for zk in Z :
        #Create new bacteria
        b = Bacterie(zk,numpy.zeros((4,4)),'n')
        Xb.append(b)

    for xk in X :
        #Lost
        if(xk.counter > cmax) :
            #really lost
            xk.etat = 'l'
            Llost.append(xk)
        else :
            #pas encore perdu
            xk.counter +=1
            Xb.append(xk)

    return Xb

def Predict(X) :
    Q = 1e-3 * numpy.identity(4)
    for b in X :
        b.P += Q

def Update(bacteria,Z) : 
    # bacteria = class and Z  = bounding box corresponding to bacteria
    R = numpy.identity(4)*0 # Uncertainty of mesurement
    K = asmatrix(bacteria.P) * asmatrix(numpy.linalg.inv(bacteria.P + R))
    bacteria.bb =  numpy.transpose(dot(K, numpy.transpose((asmatrix(Z)-asmatrix(bacteria.bb))))).tolist()
    bacteria.bb = bacteria.bb[0]
    bacteria.P =  dot((numpy.identity(4) - K),bacteria.P)

def run() :
    path_to_labels = "./datasets/500IMGS/labels/test/"
    path_to_img    = "./datasets/500IMGS/images/test/"
    path_to_out    = "./datasets/500IMGS/images/track/"

    for i,file in enumerate(os.listdir(path_to_labels)) : #for each frame 
        name,ext = os.path.splitext(file) 
        img = cv2.imread(path_to_img+name+".jpg")
        imgw, imgh = img.shape[:2]
        img_out = img.copy()
        Z = []
        with open(path_to_labels+file, "r") as labelfile:
            for line in labelfile :
                t = line.split("\n")[0].split(" ")
                [x,y,w,h] = [float(t[0]),float(t[1]),float(t[2]),float(t[3])]
                [x,y,w,h] = [int(x*imgw),int(y*imgh),int(w*imgw) ,int(h*imgh)] #Normalize
                Z.append([x,y,w,h])
        
        if i == 0 :
            print(Z)
            return
            X = []
            for bb in Z :
                X.append(Bacterie(bb, numpy.zeros((4,4)),'n'))
        else :
            Predict(X)
            X = Compare(X,Z)

        for bacteria in X :
            [x,y,w,h] =  [bacteria.bb[0],bacteria.bb[1],bacteria.bb[2],bacteria.bb[3]]
            [x,y,w,h] =  [int(x),int(y),int(w),int(h)]

            if bacteria.etat == 'n' :
                color =(0,0,255)
            else :
                color =(255,0,0)
            
            cv2.rectangle(img_out, (x-int(w/2), y-int(h/2)), (x + int(w/2), y + int(h/2)), color, 5)
        cv2.imwrite(path_to_out+name+".jpg", img_out)
        
        

if __name__ == "__main__":
    run()