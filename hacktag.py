#Importing required libraries
import playsound               
from imutils import face_utils 
from threading import Thread
import subprocess
import numpy as np
import imutils
import time
import dlib
import cv2 as cv

cap = cv.VideoCapture(0)

detector = dlib.get_frontal_face_detector() #pre trained face detector model
predictor = dlib.shape_predictor("landmarks_data.dat") #face landmarks data

#Eye aspect ratio 
def EAR(eye):
    num = np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])
    denom = np.linalg.norm(eye[0]-eye[3])*2.0
    ear = num/denom
    return ear

sound = 0
#Function to play the alarm sound while alarm is on
def sound(path):
    while(sound):
        playsound.playsound(path) 

counter = 0 #To count number of frames
alarm = 0 #Whether to play the alarm sound or not
while True:
    isTrue,frame = cap.read()
    if isTrue == 0:
        break 
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray) #Detecting faces in gray image of the current frame
    for face in faces:

        #Finding landmarks for right and left eye
        landmarks = predictor(gray,face) 
        landmarks = face_utils.shape_to_np(landmarks)
        left_eye = np.array(landmarks[36:42])
        right_eye = np.array(landmarks[42:48])

        #Calculating EAR for left and right eye and average EAR
        leftEAR = EAR(left_eye)
        rightEAR = EAR(right_eye)
        ear = (leftEAR+rightEAR)/2.0

        #Drawing contours over eyes
        leftHull =  cv.convexHull(left_eye)
        rightHull =  cv.convexHull(right_eye)
        cv.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        #If EAR is less than threshold value then eyes are closed
        if (ear<0.27):
            counter+=1
            if(counter>=48):    #If eyes are closed for 48 frames or more

                #Making a thread object and starting it to play the sound if not aldready playing
                if(alarm == 0):
                    alarm = 1
                    t = Thread(target=sound,args = ('beep.wav',)) #Replace beep.wav with whatever sound you have saved
                    sound = 1
                    t.start()

                cv.putText(frame, "DROWSINESS ALERT!", (10, 30),
				cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            #End the sound thread if eyes are no longer closed
            if(alarm==1):
                sound = 0
                t.join()
            counter = 0
            alarm = 0
            
    cv.imshow('Video', frame)
    key = cv.waitKey(1)
    if key == 27:
      	break          

cap.release()
cv.destroyAllWindows()