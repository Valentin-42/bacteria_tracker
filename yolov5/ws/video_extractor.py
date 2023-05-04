import cv2
import os
import random

def image_cutter(image_folder, output_path_sub, base_name) :
    img_num = 0

    l = os.listdir(image_folder)
    print(len(l))


    for i in range(0,len(l),30) :
        frame = l[i]
        name,ext = os.path.splitext(frame)

        if ext == ".jpg" :
            frame = cv2.imread(image_folder+"/"+frame)

            # Obtenir les dimensions de l'image
            h, w = frame.shape[:2]
            # Déterminer la différence entre la hauteur et la largeur
            d  = frame.shape[1] % 256 
            # Ajouter des bordures noires pour rendre l'image carrée
            if  d != 0 :
                frame = cv2.copyMakeBorder(frame, 0, 0, 0, 256-d, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            d = frame.shape[0] % 256 
            if  d != 0 :
                frame = cv2.copyMakeBorder(frame, 0, 256-d, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Découper l'image en plusieurs images de taille 256x256
            step = 256
            for i in range(0, h, step):
                for j in range(0, w, step):
                    sub_frame = frame[i:i+step, j:j+step]
                    cv2.imwrite(output_path_sub+"/sub"+"_"+str(i//256)+"_"+str(j//256)+"_"+base_name+"_"+name+".jpg", sub_frame)

            img_num+=1
            print("Img : "+str(name)+ " num : "+str(img_num))


def VideoExtractor(start_frame,input_video,output_path,output_path_sub,max_images) :

    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    img_num=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print("¨Problem or end of video : "+input_video)
            break
        
        # Obtenir les dimensions de l'image
        h, w = frame.shape[:2]
        # Déterminer la différence entre la hauteur et la largeur
        d  = frame.shape[1] % 256 
        # Ajouter des bordures noires pour rendre l'image carrée
        if  d != 0 :
            frame = cv2.copyMakeBorder(frame, 0, 0, 0, 256-d, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        d = frame.shape[0] % 256 
        if  d != 0 :
            frame = cv2.copyMakeBorder(frame, 0, 256-d, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        # Découper l'image en plusieurs images de taille 256x256
        if(output_path_sub != -1) :
            step = 256
            for i in range(0, h, step):
                for j in range(0, w, step):
                    sub_frame = frame[i:i+step, j:j+step]
                    cv2.imwrite(output_path_sub+"sub"+str(img_num)+"_{}_{}.jpg".format(i//256, j//256), sub_frame)

        cv2.imwrite(output_path+'img'+str(img_num)+'.jpg',frame)
        img_num+=1
        print(str(img_num)+" over "+str(max_images))
        if(max_images == -1) :
            continue
        if(img_num == max_images) :
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    
    
    image_folder = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/raw/Manip2-debut_raw"
    output_path_sub = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/raw/Manip2-debut_raw/sub"
    video_path = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/Videos/"
    
    # for video_name in os.listdir(video_path) :
    #     if video_name.endswith('.m4v'):
    #         filename, extension = os.path.splitext(video_name)
    #         if not os.path.exists(video_path+filename+"/"):
    #             os.makedirs(video_path+filename+"/")
            
    #         VideoExtractor(0,video_path+video_name,video_path+filename+"/",-1,-1)
    
    for video_name in os.listdir(video_path) :
        filename, extension = os.path.splitext(video_name)
        if extension == '.m4v':

            folder_img_path = video_path+filename+"/"
            folder_sub_path = folder_img_path+"sub/"

            if not os.path.exists(folder_sub_path):
                os.makedirs(folder_sub_path)

            image_cutter(folder_img_path,folder_sub_path,filename)