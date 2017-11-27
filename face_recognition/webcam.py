# import site;
# site.getsitepackages();
import cv2
import sys 
from datetime import *
import numpy as np
from PIL import Image
import random
import os
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

count=0
faceWidth=30
faceHeight=30
facesData = []
while count < 100:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(90, 90),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    #print gray
    print "Found {0} faces!".format(len(faces))
    # Draw a rectangle around the faces
    
    if len(faces) > 0:
        (x, y, w, h) = faces[len(faces) - 1]
        count+=1
        print count
        print (x,y,w,h)
        if count%10 == 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceFrame = frame[y:y+h, x:x+w]
            #count2 = str(count)
            #cv2.imwrite("facesData/5/"+count2+'.jpg', faceFrame)
            cv2.imwrite("face.jpg", faceFrame)#save a temp image
            faceImg = Image.open("face.jpg")#read the temp image
            faceImg = faceImg.resize((faceWidth, faceHeight),Image.BILINEAR)
            faceImg = faceImg.convert("L")
            faceImg = np.reshape(faceImg, (1, faceWidth*faceHeight))#reshape the array
            #print faceImg[0]
            facesData.append(faceImg[0])
                
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
facesData = np.array(facesData)

print facesData
facesDataFile = open('facesData/facesData1.npy', 'w')
np.save(facesDataFile, facesData)
facesDataFile.close()

#print os.path.isfile('facesData/facesData7.npy')
#faceDataFile = open('facesData/facesData2.npy', 'r')
#faceDataFile2 = open('facesData/facesData2.npy', 'r')
#x = faceDataFile.readlines()
#print x 
#if len(x) > 0:
    #faces = np.load(faceDataFile2)
    #print faces
#faceDataFile.close()
#faceDataFile2.close()
    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
