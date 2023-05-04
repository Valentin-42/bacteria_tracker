import cv2
import numpy as np
import os
import random

class DatasetCreator :

    def __init__(self, path_to_sub_images,path_to_ds, makedirs) :
        self.path_to_sub_images = path_to_sub_images
        self.path_to_ds  = path_to_ds
        self.valid_images_ext = [".jpg",".png"]
        
        if not os.path.exists(self.path_to_ds) and makedirs:
            self.create_folder_architecture()
            
    def create_folder_architecture(self):

        os.makedirs(self.path_to_ds+"/images/train")
        os.makedirs(self.path_to_ds+"/images/val")
        os.makedirs(self.path_to_ds+"/images/test")
        os.makedirs(self.path_to_ds+"/images/groundtruths")

        os.makedirs(self.path_to_ds+"/labels/train")
        os.makedirs(self.path_to_ds+"/labels/val")
        os.makedirs(self.path_to_ds+"/labels/test")

    def check_treashold(self) :
        
        img_name = os.listdir(self.path_to_raw_images)[0]

        img = cv2.imread(self.path_to_raw_images+'/'+img_name)
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_limit = np.array([0, 0, 30])
        upper_limit = np.array([179, 255, 255])
        mask = cv2.inRange(img_HSV, lower_limit, upper_limit)
        img_Filtered = cv2.bitwise_and(img,img,mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours : 
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_Filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)

        dimg = cv2.resize(img_Filtered, (round(mask.shape[1] / 4), round(mask.shape[0] / 4)))
        oi =   cv2.resize(img, (round(mask.shape[1] / 4), round(mask.shape[0] / 4)))

        cv2.imwrite('img_originale.jpg',img)
        cv2.imwrite('mask.jpg',mask)
        print("done")

        # cv2.imshow("Labeled", dimg)
        # cv2.imshow("Origin", oi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def set_path_to_sub_images(self, path): 
        self.path_to_sub_images = path

    def label_folder(self, dest, N_fill, N_empty) :
        
        cnt_empty = 0
        cnt_fill = 0
        #Create folder
        if not os.path.exists(self.path_to_ds):
            self.create_folder_architecture()

        img_list = os.listdir(self.path_to_sub_images)
        random.shuffle(img_list)

        for f in  img_list :
            name,ext = os.path.splitext(f)
            print(name)

            if ext.lower() not in self.valid_images_ext:
                print("pass")
                continue
            
            f = self.path_to_sub_images+"/"+f
            img = cv2.imread(f)
            imgo = img.copy()

            img_h, img_w = img.shape[:2]
            img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_limit = np.array([0, 0, 40])
            upper_limit = np.array([179, 255, 255])

            img_HSV[:,0,:] = cv2.equalizeHist(img_HSV[:,0,:])
            mask = cv2.inRange(img_HSV, lower_limit, upper_limit)
            mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2)
            img_Filtered = cv2.bitwise_and(img,img,mask=mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            with open(self.path_to_ds+"labels/"+dest+name+".txt", "w") as file:
                empty = True
                for i,c in enumerate(contours) : 
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    file.write(f"0 {(2*x+w)/(2*img_w)} {(2*y+h)/(2*img_h)} {w/img_w} {h/img_h}\n")
                    empty = False

                if (empty == True and cnt_empty<N_empty) :
                    cv2.imwrite(self.path_to_ds+"images/groundtruths/"+name+".jpg", img)
                    cv2.imwrite(self.path_to_ds+"images/"+dest+name+".jpg", imgo)
                    cnt_empty += 1

                if (empty == False and cnt_fill<N_fill) :
                    cv2.imwrite(self.path_to_ds+"images/groundtruths/"+name+".jpg", img)
                    cv2.imwrite(self.path_to_ds+"images/"+dest+name+".jpg", imgo)
                    cnt_fill += 1

                if (cnt_fill >= N_fill and cnt_empty >= N_empty) :
                    break

            print("Total of "+str(cnt_fill)+" fill images and "+str(cnt_empty)+" empty images")

    def cleaner(self, folder_with_files_to_delete,reference_folder) :

        # get a list of filenames (without extension) in the reference folder
        reference_filenames = [os.path.splitext(f)[0] for f in os.listdir(reference_folder)]

        # loop through the files in the folder to delete
        for filename in os.listdir(folder_with_files_to_delete):
            # remove the extension from the filename
            filename_without_ext = os.path.splitext(filename)[0]
            # if the filename (without extension) is not in the reference folder, delete the file
            if filename_without_ext not in reference_filenames:
                os.remove(os.path.join(folder_with_files_to_delete, filename))

        print(len(os.listdir(reference_folder)), len(os.listdir(folder_with_files_to_delete)))



def create_dataset_with_treashold_annotation(video_path) :
    sum = 0
    for video_name in os.listdir(video_path) :
        filename, extension = os.path.splitext(video_name)
        if extension == '.m4v':
            folder_img_path = video_path+filename+"/"
            # Raw SubImages finishing by /
            folder_sub_path = folder_img_path+"sub/"
            #Switching btw multiple folders
            DSC.set_path_to_sub_images(folder_sub_path)
            nbs = len(os.listdir(folder_sub_path))
            N_empty =  int(nbs*0.001)
            N_fill  =  int(nbs*0.02)
            sum +=  N_empty+N_fill
            print("taking e: "+str(N_empty) +" f: "+ str(N_fill) +" from "+ str(filename))
            DSC.label_folder(dest="train/",N_empty=N_empty, N_fill=N_fill)

    nbs = len(os.listdir(path_to_dataset+"images/train/"))
    print(str(sum) +" over "+ str(nbs))


if __name__ == "__main__":

    # Creating a YOLOv5 dataset structure - Please finish paths by '/'
    path_to_dataset = "~/local_storage/bacteria_tracker/datasets/full_dataset/"

    path_to_sub_images = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/full_dataset/images/"

    DSC = DatasetCreator("",path_to_dataset,True)


    folder_with_files_to_delete = path_to_dataset +"images/val/"
    reference_folder  = path_to_dataset +"labels/val/"
    DSC.cleaner(folder_with_files_to_delete,reference_folder)

    # video_path = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/Videos/"
    # create_dataset_with_treashold_annotation(video_path)
    
    


