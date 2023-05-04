import cv2
import os
import sys

def extract_number(file_name):
    return int(file_name.split('img')[1].split('.jpg')[0])

def extract_number2(file_name,video_ouput_path):
    return int(file_name.split('Adh_')[1].split('.jpg')[0])

def image_to_video(path,video_ouput_path,video_name) :
    
    # Liste des noms de fichiers dans le dossier
    file_names = sorted([os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith('.jpg')], key=extract_number)
    # Lecture de la première image pour déterminer les dimensions de la vidéo
    img = cv2.imread(file_names[0])
    height, width, layers = img.shape

    # Initialisation de la vidéo
    video = cv2.VideoWriter(video_ouput_path+video_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))

    # Parcours des images et écriture dans la vidéo
    for file_name in file_names:
        img = cv2.imread(file_name)
        video.write(img)
        print(file_name)

    # Fermeture de la vidéo
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":

    # Chemin du dossier contenant les images
    # path = "../runs/detect/bacteria_raw_yolov5s_500IMGLB"   #'/chemin/vers/le/dossier/contenant/les/images'
    #path = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/experiment/"    #'/chemin/vers/le/dossier/contenant/les/images'
    
    path = sys.argv[1]
    video_ouput_path = sys.argv[2] #"./video/"
    video_name = "video.avi"
    image_to_video(path,video_ouput_path,video_name)
