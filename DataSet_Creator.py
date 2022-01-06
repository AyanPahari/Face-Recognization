import cv2
import numpy as np
import sqlite3

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

def insertOrUpdate(Id,Name):
    conn=sqlite3.connect('FaceBase.db')
    cmd='SELECT * FROM people WHERE ID= '+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for _ in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd='UPDATE people SET Name= '+str(Name)+'WHERE ID= '+str(Id)
    else:
        cmd='INSERT INTO people(ID,Name) VALUES('+str(Id)+','+str(Name)+')'
    conn.execute(cmd)
    conn.commit()
    conn.close()
                         
id = input('Enter User Id: ')
name=input('Enter your name: ')
insertOrUpdate(id,name)
sampleNum=0
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        sampleNum+=1
        cv2.imwrite("Dataset1/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow('Face_Detect',frame)
    cv2.waitKey(1)
    if sampleNum>50:
        break
cap.release()
cv2.destroyAllWindows()
