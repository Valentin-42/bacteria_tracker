import cv2
import numpy
import os

class Tracker :

    def __init__(self, path_to_ds) :
        #Images with detections
        self.test_images  = self.path_to_ds+"/images/test" 
        self.test_label   = self.path_to_ds+"/labels/test" 
        #Images with applied tracking
        self.track_label  = self.path_to_ds+"/labels/track/" #Where we store tracked bacterias
        self.track_images  = self.path_to_ds+"/images/track/"#Where we plot colored bacterias 

        self.valid_images_ext = [".jpg",".png"]
        self.overlap_threashold = 0.8 #ie overlap btw 2 bacteria bb must be >80%
        self.colormap = ['#37AB65', '#3DF735'] #Random colors for now, must be changed to a progressive color map
    

    def init_tracking(self):
        first_label = os.listdir(self.test_label)[0]
        with open(self.test_label+first_label, "r") as file :
            for line in file1:
                t = line.split(" ")
                if(len(t) == 5) :
                    #Add match arg to 0
                    t.append(0)
        with open(self.track_label+first_label, "w") as file :
            file.write(f"{t[0]} {t[1]} {t[2]} {t[3]} {t[4]} {t[5]}\n")

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

    def bacteria_tables_extraction(self, file1_name, file2_name) :
        t1 = []
        t2 = []
        with open(file1_name, "r") as file1, open(file2_name, "r") as file2:
            for line in file1:
                t1.append(line.split(" "))
            for line in file2 :
                t2.append(line.split(" "))
        return t1,t2


    def tracking_method1(self) :
        for i,file in enumerate(os.listdir(self.test_label)[1:]) : #for image i, i>0
            name,ext = os.path.splitext(os.listdir(self.test_label)[i+1]) # Create a new label file for image i+1
            with open(self.track_label+name+".txt", "a+") as file:
                t1,t2 = self.bacteria_tables_extraction(os.listdir(self.track_label)[i-1],file)
                for bacteria in t1 :
                    [c1,x1,y1,w1,h1,m1] = bacteria
                    for index, bacteria in enumerate(t2) : 
                        [c2,x2,y2,w2,h2] = bacteria
                        overlap_score = self.compute_iou([x1,y1,2*w1,2*h1],[x2,y2,w2,h2]) #Compute overlap between this one and the other one
                        if(overlap_score > self.overlap_threashold) :
                            # It is the same bacteria -> Change color and match them
                            # Match <=> Added to file and pop ; Color <=> +1 to arg match
                            file.write(f"{c2} {x2} {y2} {h2} {m1+1}\n")
                            t2.pop(index)


if __name__ == "__main__":
    path_to_ds = "./datasets/500IMGS/" 
    t = Tracker(path_to_ds)
    t.init_tracking()
    t.tracking_method1()