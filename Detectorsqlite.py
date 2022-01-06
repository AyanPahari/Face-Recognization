import cv2
import numpy as np
import sqlite3

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('Recognizer\\trainingdata1.yml')
def getProfile(id):
    conn=sqlite3.connect('FaceBase.db')
    cmd='SELECT * FROM people WHERE ID= '+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
        
id=0
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print(conf)
        profile=getProfile(id)
        if profile!=None:
            if conf<55:
                cv2.putText(frame,"Name: "+str(profile[1]),(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                cv2.putText(frame,"Age: "+str(profile[2]),(x,y+h+40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                cv2.putText(frame,"Gender: "+str(profile[3]),(x,y+h+60),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                cv2.putText(frame,"Branch: "+str(profile[4]),(x,y+h+80),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

            else:
                cv2.putText(frame,'Unknown',(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
    cv2.imshow('Face_Detect',frame)
    if(cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
