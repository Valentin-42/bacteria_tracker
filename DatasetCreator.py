import cv2
import numpy as np
import os

class DatasetCreator :

    def __init__(self, path_to_raw_images,path_to_ds) :

        self.path_to_raw_images = path_to_raw_images
        self.path_to_ds  = path_to_ds
        self.valid_images_ext = [".jpg",".png"]
        
        if not os.path.exists(self.path_to_ds):
            self.CreateFolderArchitecture()
            

    def CreateFolderArchitecture(self):

        os.makedirs(self.path_to_ds+"/images/train")
        os.makedirs(self.path_to_ds+"/images/val")
        os.makedirs(self.path_to_ds+"/images/test")
        os.makedirs(self.path_to_ds+"/images/groundtruths")

        os.makedirs(self.path_to_ds+"/labels/train")
        os.makedirs(self.path_to_ds+"/labels/val")
        os.makedirs(self.path_to_ds+"/labels/test")

    def CheckTreashold(self, index) :
       
        img = cv2.imread(self.path_to_raw_images+"/img"+str(index)+".jpg")
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_limit = np.array([0, 0, 90])
        upper_limit = np.array([179, 255, 255])
        mask = cv2.inRange(img_HSV, lower_limit, upper_limit)
        img_Filtered = cv2.bitwise_and(img,img,mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours : 
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_Filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)

        dimg = cv2.resize(img_Filtered, (round(mask.shape[1] / 4), round(mask.shape[0] / 4)))
        oi =   cv2.resize(img, (round(mask.shape[1] / 4), round(mask.shape[0] / 4)))

        cv2.imshow("Labeled", dimg)
        cv2.imshow("Origin", oi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def LabelFolder(self, dest) :
        
        #Create folder
        if not os.path.exists(self.path_to_ds):
            self.CreateFolderArchitecture()


        for f in os.listdir(self.path_to_raw_images) :
            name,ext = os.path.splitext(f)
            print(name)

            if ext.lower() not in self.valid_images_ext:
                print("pass")
                continue
            
            f = self.path_to_raw_images+"/"+f
            img = cv2.imread(f)
            img_h, img_w = img.shape[:2]
            img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_limit = np.array([0, 0, 80])
            upper_limit = np.array([179, 255, 255])
            mask = cv2.inRange(img_HSV, lower_limit, upper_limit)
            img_Filtered = cv2.bitwise_and(img,img,mask=mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            with open(self.path_to_ds+"labels/"+dest+name+".txt", "w") as file:
                
                for i,c in enumerate(contours) : 
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(img_Filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    file.write(f"0 {(2*x+w)/(2*img_w)} {(2*y+h)/(2*img_h)} {w/img_w} {h/img_h}\n")
            
            cv2.imwrite(self.path_to_ds+"images/"+dest+name+".jpg", img)
            cv2.imwrite(self.path_to_ds+"images/groundtruths/"+name+".jpg", img_Filtered)



if __name__ == "__main__":

    # Raw Images
    path_to_images_to_label = "raw/sub_images/"
    # Creating a YOLOv5 dataset structure
    path_to_dataset = "datasets/10IMGS/"

    DSC = DatasetCreator(path_to_images_to_label,path_to_dataset)
    DSC.LabelFolder(dest="train/") #Everything to training 

    #L.CheckTreashold(3)
