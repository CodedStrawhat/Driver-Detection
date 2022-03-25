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

s = 0
#Function to play the alarm sound while alarm is on
def sound(path):
    while(s):
        playsound.playsound(path) 

counter = 0 #To count number of frames
alarm = 0 #Whether to play the alarm sound or not
a = 0
while True:

    isTrue,frame = cap.read()
    if isTrue == 0:
        break 

    rgb_im = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Converting image to rgb

    #Finding landmarks for right and left eye
    output = face_model.process(rgb_im)
    output = output.multi_face_landmarks[0]
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

    #Calculating EAR for left and right eye and average EAR
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
        counter+=1
        if(counter>=48):    #If eyes are closed for 48 frames or more

            #Making a thread object and starting it to play the sound if not aldready playing
            if(alarm == 0):
                alarm = 1
                t = Thread(target=sound,args = ('beep.wav',)) #Replace beep.wav with whatever sound you have saved
                s = 1
                t.start()

            cv.putText(frame, "DROWSINESS ALERT!", (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        #End the sound thread if eyes are no longer closed
        if(alarm==1):
            s = 0
            t.join()
        counter = 0
        alarm = 0
    cv.imshow('Video', frame)
    key = cv.waitKey(1)
    if key == 27:
      	break          

cap.release()
cv.destroyAllWindows()
