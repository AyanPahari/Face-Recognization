import cv2
import numpy as np
import urllib.request as ur

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('Recognizer\\trainingdata.yml')
url='http://192.168.43.1:8080/shot.jpg'
id=0
while True:
    imgResp=ur.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print(conf)
        if id==1:name='ANKIT'
        elif id==2:name='AYAN'
        if conf<50:
            cv2.putText(img,name,(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        else:
            cv2.putText(img,'Unknown',(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
    imgr=cv2.resize(img,(640,480))
    cv2.imshow('Face_Detect',imgr)
    if(cv2.waitKey(1)==ord('q')):
        break
cv2.destroyAllWindows()
