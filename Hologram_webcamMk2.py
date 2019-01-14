import cv2
import numpy as np
import os,sys
import pyscreenshot as ImageGrab

#Get the face_cascade
face_cascade = cv2.CascadeClassifier('/home/otoshuki/anaconda/envs/tensorflow/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

#Passing function for the trackbars
def nothing(x):
    pass

#Create window and Trackbars
cv2.namedWindow('RGB')

cv2.createTrackbar('R-high', 'RGB', 255, 255, nothing)
cv2.createTrackbar('G-high', 'RGB', 255, 255, nothing)
cv2.createTrackbar('B-high', 'RGB', 255, 255, nothing)

cap = cv2.VideoCapture(0)

#The function to create the hologram image
def makeHologram(original,scale=1.5,scaleR=3,distance=0):
    '''
        Create 3D hologram from image (must have equal dimensions)
    '''

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

    #cv2.imshow("up",up)
    #cv2.imshow("down",down)
    #cv2.imshow("left",left)
    #cv2.imshow("right",right)
    cv2.imshow("hologram",hologram)
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

#######################################
#Main Loop
while(1):

    #Screen Record but returns a bluish image
    ret, img_read = cap.read()
    img_np = np.array(img_read)

    mask = np.zeros(img_read.shape[:2], np.uint8)
    bdModel = np.zeros((1,65), np.float64)
    fdModel = np.zeros((1,65), np.float64)

    #Converting to RGB
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    #Getting trackbar values
    rr = int(cv2.getTrackbarPos('R-high', 'RGB'))
    gg = int(cv2.getTrackbarPos('G-high', 'RGB'))
    bb = int(cv2.getTrackbarPos('B-high', 'RGB'))

    #Setting a specific points to get the lower threshold
    cv2.rectangle(img,(110,100),(114,104),(0,0,0),1)
    r1,g1,b1 = img[52,52]
    cv2.rectangle(img, (110,200),(114,204),(0,0,0),1)
    r2,g2,b2 = img[52,302]
    cv2.rectangle(img, (470,100),(474,104),(0,0,0),1)
    r3,g3,b3 = img[472,52]
    cv2.rectangle(img, (470,200),(474,204),(0,0,0),1)
    r4,g4,b4 = img[472,302]
    rav = int((r1+r2+r3+r4)/2)
    gav = int((g1+g2+g3+g4)/2)
    bav = int((b1+b2+b3+b4)/2)

    #Take lower and upper limits and make the mask
    back_lower = np.array([rav,gav,bav])
    back_upper = np.array([rr,gg,bb])
    mask = cv2.inRange(img, back_lower, back_upper)
    mask_inv = cv2.bitwise_not(mask)

    #Morphological Transformations
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    blur1 = cv2.medianBlur(mask_inv, 5)
    kernel2 = np.ones((5,5),np.uint8)
    closing1 = cv2.morphologyEx(blur1, cv2.MORPH_CLOSE, kernel2)
    blur2 = cv2.medianBlur(closing1, 5)
    closing2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel1)
    dilate = cv2.dilate(closing2, kernel1, iterations = 1)

    #Getting the foreground
    fore = cv2.bitwise_and(img,img,mask = closing2)

    #Detect the face and crop it
    faces = face_cascade.detectMultiScale(img_read, 1.3, 5)

    for (x,y,w,h) in faces:
        sub_face = fore[y:y+220, x:x+220]
        rect = [x,y,x+w,y+h]
    cv2.grabCut(img_np,rect,bdModel,fdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0),0,1).astype('uint8')
    new = img_np*mask2[:,:,np.newaxis]

    cv2.imshow('New',new)
    #Make the hologram using the function and display it
    #makeHologram(sub_face.copy())

    #Show the cropped image
    #cv2.imshow('Cropped', sub_face)

    cv2.imshow('image', img_read)

    #Wait for the ESC key to be pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
