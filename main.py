#Electronics Club - IITG
#Guining Pertin - Otoshuki
#Import libraries
import cv2
import numpy as np
import os,sys

#Passing function for the trackbars
def nothing(x):
    pass

def run():
    #Get the face_cascade
    face_cascade = cv2.CascadeClassifier('/home/otoshuki/anaconda/envs/tensorflow/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #Create window and Trackbars
    # cv2.namedWindow('RGB')
    # cv2.createTrackbar('R-high', 'RGB', 255, 255, nothing)
    # cv2.createTrackbar('G-high', 'RGB', 255, 255, nothing)
    # cv2.createTrackbar('B-high', 'RGB', 255, 255, nothing)
    #VideoCapture instance
    cap = cv2.VideoCapture(0)
    #Main program
    while True:
        _, frame = cap.read()
        #Detect face
        #Crop face
        #Background subtraction
        #Fading effects
        #Display
        #3D warp
        #3D Reconstruction
        #Show results
        cv2.imshow('Input', frame)
        #Wait for ESC to be pressed
        key = cv2.waitKey(5) & 0xFF
        if key == 27: break
#Run
if __name__ == "__main__":
    run()
