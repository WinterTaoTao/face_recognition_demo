from PIL import Image
import glob
import numpy as np
#import KnnClassification as KC
import cv2
import sys 
from datetime import *
from PIL import Image
import random

import sys
sys.path.append('face_recognition')
from faceRecognition import FR

video_capture = cv2.VideoCapture(0)
cascPath = "face_recognition/haarcascade_frontalface_default.xml"
facesDataPath = "face_recognition/facesData"

faceRec = FR(cascPath, facesDataPath)

#read images from faces datasets
faceRec.readImages()
count=[0] #number of total recognised face
cnt1=[0] #number of total recognised face1
cnt2=[0] #number of total recognised face2
cnt3=[0] #number of total recognised face3
cnt4=[0] #number of total recognised face4
rejectCount=[0] #number of total recognised faces which are not in data sets

noFaceCount=0 #number of no face times

lastFaceQuantity=0 #number of faces in the last frame

while True:
    ret, frame=video_capture.read()
    cls = faceRec.recogniseFaces(frame)
    if len(cls) == 0:
        noFaceCount+=1
        if(noFaceCount >= 10):
            count=[0]
            cnt1=[0]
            cnt2=[0]
            cnt3=[0]
            cnt4=[0]
            rejectCount=[0]
            noFaceCount=0
    if len(cls) > 0:
        faceQuantity = len(cls)
        if faceQuantity < lastFaceQuantity:
            count=[0]
            cnt1=[0]
            cnt2=[0]
            cnt3=[0]
            cnt4=[0]
            rejectCount=[0]
            noFaceCount=0
        elif faceQuantity > lastFaceQuantity:
            count.append(0)
            cnt1.append(0)
            cnt2.append(0)
            cnt3.append(0)
            cnt4.append(0)
            rejectCount.append(0)
        
        people = []
        for i in range(0, faceQuantity):
            #increase count
            count[i]+=1
            if cls[faceQuantity-1-i]==1:
                print "Found Juntao!"
                cnt1[i]+=1
            elif cls[faceQuantity-1-i]==2:
                print "Found Wentao!"
                cnt2[i]+=1
            elif cls[faceQuantity-1-i]==3:
                print "Found Xiaotong"
                cnt3[i]+=1
            elif cls[faceQuantity-1-i]==4:
                print "Found Yi"
                cnt4[i]+=1
            elif cls[faceQuantity-1-i]=="reject":
                print "Reject"
                rejectCount[i]+=1
            
        
        
            #The possibilities for the face to belong to a person
            pos = {}
            pos["Juntao"] = (cnt1[i]+0.0)/count[i]
            pos["Wentao"] = (cnt2[i]+0.0)/count[i]
            pos["Xiaotong"] = (cnt3[i]+0.0)/count[i]
            pos["Yi"] = (cnt4[i]+0.0)/count[i]
            pos["Reject"]=(rejectCount[i]+0.0)/count[i]
            maxName = ''#name with max possibility
            maxPos = 0.0
            print 'Face' + str(i)
            print pos
        
        
            for (k,v) in pos.items():
                if v > maxPos:
                    maxName = k
                    maxPos = v
            #if maxName == 'Reject':
                #print 'Unregister'
            #else:
                #print 'Hello! ' + maxName + '! '
            people.append(maxName)
        print people
                
    cv2.imshow('Vedio', frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
