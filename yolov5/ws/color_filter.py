import cv2
import numpy as np

path = "/home/GPU/vvial/home_gtl/bacteria_tracker_ws/raw/Manip2-debut_contrast/Adh_0403.jpg"

#Defining window
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 500)

cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, lambda *args: None)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, lambda *args: None)

cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, lambda *args: None)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, lambda *args: None)

cv2.createTrackbar("Value Min", "TrackBars", 0, 255, lambda *args: None)
cv2.createTrackbar("Value Max", "TrackBars", 255, 255, lambda *args: None)

cv2.createTrackbar("Alpha", "TrackBars", 10, 255, lambda *args: None)
cv2.createTrackbar("Beta", "TrackBars", 0, 100, lambda *args: None)

img = cv2.imread(path)
img = cv2.resize(img, (round(img.shape[1] / 4), round(img.shape[0] / 4)))

while True:

    hue_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackBars")

    sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")

    val_min = cv2.getTrackbarPos("Value Min", "TrackBars")
    val_max = cv2.getTrackbarPos("Value Max", "TrackBars")

    alpha = cv2.getTrackbarPos("Alpha", "TrackBars")
    beta = cv2.getTrackbarPos("Beta", "TrackBars")

    imgC = img.copy()
    imgC = cv2.convertScaleAbs(img, alpha=alpha, beta=-beta)

    img_HSV = cv2.cvtColor(imgC, cv2.COLOR_BGR2HSV)

    lower_limit = np.array([hue_min, sat_min, val_min])
    upper_limit = np.array([hue_max, sat_max, val_max])

    mask = cv2.inRange(imgC, lower_limit, upper_limit)

    img_Filtered = cv2.bitwise_and(img,img,mask=mask)

    # Display
    cv2.imshow("Image O", img)
    cv2.imshow("Image C", imgC)
    cv2.imshow("Image HSV", img_HSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtre", img_Filtered)

    cv2.waitKey(1)

