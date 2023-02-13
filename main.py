import cv2
import numpy

class Model :

    def __init__(self, path_to_ds) :
        self.path_to_ds = "./datasets/10IMGS/" 

        self.train_images = self.path_to_ds+"/images/train" 
        self.val_images   = self.path_to_ds+"/images/val" 
        self.train_label  = self.path_to_ds+"/labels/train" 
        self.val_label    = self.path_to_ds+"/labels/val" 

        self.valid_images_ext = [".jpg",".png"]

    def Run_Model() :
        l = """
python train.py --img 256 --batch 16 --epochs 3 --data IMGS10.yaml --weights yolov5s.pt
        """
        print("Make sure there is the .yaml file in the yolov5/data !")
        print("Training cmd : ")
        print("\n")
        print(l)


f = "raw/images/img0.jpg"
img = cv2.imread(f)
img = cv2.resize(img, (round(img.shape[1] / 4), round(img.shape[0] / 4)))

img_h, img_w = img.shape[:2]
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img1 = img.copy()
img2 = img.copy()
img3 = img.copy()

img1 = cv2.convertScaleAbs(img1, alpha=10, beta=0)


cv2.imshow("Image C", img1)
cv2.imshow("Image O", img2)
cv2.waitKey(0)