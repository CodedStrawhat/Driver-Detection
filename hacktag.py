#Importing required libraries
import playsound               
from threading import Thread
import numpy as np
import cv2 as cv
import mediapipe as mp
from scipy.spatial import distance as dis

cap = cv.VideoCapture(0)

#Mediapipe's face classification model
face_mesh = mp.solutions.face_mesh
face_model = face_mesh.FaceMesh(static_image_mode = False,refine_landmarks = True)

#Eye aspect ratio 
def EAR(eye):
    num = dis.euclidean(eye[2],eye[3])
    denom = dis.euclidean(eye[0],eye[1])
    ear = num/denom
    return ear

#Lip aspect ratio
def LAR(vert,horz):
    num = dis.euclidean(horz[0],horz[1])
    denom = dis.euclidean(vert[0],vert[1])
    ear = num/denom
    return ear

s = 0
#Function to play the alarm sound while alarm is on
def sound(path):
    while(s):
        playsound.playsound(path) 

counter_eye = 0 #To count number of frames
alarm = 0 #Whether to play the alarm sound or not
counter_lip = 0 
a = 0
d = np.zeros(2)
while True:

    isTrue,frame = cap.read()
    if isTrue == 0:
        break 
    rgb_im = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Converting image to rgb

    #Finding landmarks for face
    output = face_model.process(rgb_im)
    try:
        output = output.multi_face_landmarks[0]
    except:
        cv.imshow('Video', frame)
        key = cv.waitKey(1)
        if key == 27:
            break
        continue
    l = []
    for landmark in output.landmark:
        x = (landmark.x)*frame.shape[1] #Denormalizing points
        y = (landmark.y)*frame.shape[0]
        l.append([x,y])
    l = np.array(l,dtype = int)

    #Indexes of landmarks of left and right eye
    left_eye = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    right_eye = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    left_eye = np.array(l[left_eye[:]][:],dtype = int)
    right_eye = np.array(l[right_eye[:]][:],dtype = int)

    #Points required to find EAR
    left_points = np.array(l[[386,374,263,362]][:],dtype = int)
    right_points = np.array(l[[159, 145,133,33]][:], dtype = int)

    #Calculating Eye aspect ratio for left and right eye and average EAR
    leftEAR = EAR(left_points)
    rightEAR = EAR(right_points)
    ear = (leftEAR+rightEAR)/2.0

    #Drawing contours over eyes
    leftHull =  cv.convexHull(left_eye)
    rightHull =  cv.convexHull(right_eye)
    cv.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
    cv.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

    #If EAR is more than threshold value then eyes are considered closed
    if (ear>5):
        counter_eye+=1
        if(counter_eye>=60):    #If eyes are closed for 60 frames or more
            d[0] = 1
    else : 
        d[0] = 0
        counter_eye = 0
    
    #Drawing contour over lips
    lip_land = [ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ] 
    lip = np.array(l[lip_land[:]][:],dtype = int)
    lipHull = cv.convexHull(lip)
    cv.drawContours(frame, [lipHull], -1, (0,255,0), 1)

    #Indexes for top,bottom and left,right of lips
    left_right = np.array(l[[78,308]][:],dtype = int)
    top_bot = np.array(l[[13,14]][:],dtype = int)
    #Calculating lip ratio
    lip_ratio = LAR(top_bot,left_right)
    
    if(lip_ratio <1.8):
        counter_lip += 1
        if(counter_lip >= 45):
            d[1] = 1
    else:
        counter_lip = 0
        d[1] = 0
        
    if 1 in d:
        #Making a thread object and starting it to play the sound if not aldready playing
        if(alarm == 0): 
            alarm = 1
            t = Thread(target=sound,args = ('beep.wav',)) #Replace beep.wav with whatever sound you have saved
            s = 1
            t.start()

        cv.putText(frame, "DROWSINESS ALERT!", (10, 30),
        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        #End the sound thread if eyes are no longer closed and person is not yawning
        if(alarm==1):
            s = 0
            t.join()
        alarm = 0

    cv.imshow('Video', frame)
    key = cv.waitKey(1)
    if key == 27:
      	break          

cap.release()
cv.destroyAllWindows()