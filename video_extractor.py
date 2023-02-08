import cv2

def VideoExtractor(start_frame) :

    output_path = "raw/images/"
    output_path_sub = "raw/sub_images/"

    cap= cv2.VideoCapture('raw/video/bacteria.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    img_num=0
    max_images=20
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print("¨Problem")
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
        step = 256
        for i in range(0, h, step):
            for j in range(0, w, step):
                sub_frame = frame[i:i+step, j:j+step]
                cv2.imwrite(output_path_sub+"sub"+str(img_num)+"_{}_{}.jpg".format(i//256, j//256), sub_frame)

        cv2.imwrite(output_path+'img'+str(img_num)+'.jpg',frame)
        img_num+=1
        print(str(img_num)+" over "+str(max_images))
        if(img_num == max_images) :
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0




if __name__ == "__main__":
    VideoExtractor(200)