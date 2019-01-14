#Import libraries
import cv2
import numpy as np

#The function to create the hologram image
def makeHologram(original,scale=1.5,scaleR=3,distance=0):

    height = int((scale*original.shape[0]))
    width = int((scale*original.shape[1]))

    image = cv2.resize(original, (width, height), interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('Img', image)
    up = image.copy()
    down = rotate_bound(image.copy(),180)
    right = rotate_bound(image.copy(), 90)
    left = rotate_bound(image.copy(), 270)
    hologram = np.zeros([max(image.shape)*scaleR+distance,max(image.shape)*scaleR+distance,3], image.dtype)

    center_x = int((hologram.shape[0])/2)
    center_y = int((hologram.shape[1])/2)

    vert_x = int((up.shape[0])/2)
    vert_y = int((up.shape[1])/2)
    hologram[0:up.shape[0], center_x-vert_x+distance:center_x+vert_x+distance] = up
    hologram[ hologram.shape[1]-down.shape[1]:hologram.shape[1] , center_x-vert_x+distance:center_x+vert_x+distance] = down
    hori_x = int((right.shape[0])/2)
    hori_y = int((right.shape[1])/2)
    hologram[ center_x-hori_x : center_x-hori_x+right.shape[0] , hologram.shape[1]-right.shape[0]+distance : hologram.shape[1]+distance] = right
    hologram[ center_x-hori_x : center_x-hori_x+left.shape[0] , 0+distance : left.shape[0]+distance ] = left

    #cv2.imshow("Hologram",hologram)
    #cv2.waitKey()
    return hologram

#Function to rotate the image
def rotate_bound(image, angle):
    
    # grab the dimensions of the image and then determine the
    # center
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

#Functions for morphological transformations
def transform(layers,mask):
    if layers == 1:
        mask = cv2.GaussianBlur(mask,(5,5),2)
    if layers == 2:
        mask = cv2.GaussianBlur(mask,(5,5),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(4))
    if layers == 3:
        mask = cv2.GaussianBlur(mask,(5,5),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(6))
    if layers == 4:
        mask = cv2.GaussianBlur(mask,(5,5),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(6))
        mask = cv2.medianBlur(mask, 5)
    if layers == 5:
        #Older working version
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        blur1 = cv2.medianBlur(mask_inv, 5)
        kernel2 = np.ones((5,5),np.uint8)
        closing1 = cv2.morphologyEx(blur1, cv2.MORPH_CLOSE, kernel2)
        blur2 = cv2.medianBlur(closing1, 5)
        closing2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel1)
        mask = cv2.dilate(closing2, kernel1, iterations = 1)
    return mask
