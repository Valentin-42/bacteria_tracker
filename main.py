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

    
