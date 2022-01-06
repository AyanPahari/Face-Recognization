import os
import numpy as np
import cv2
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def getImageAndId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        Id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(Id)
        ids.append(Id)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return ids,faces

Ids,faces=getImageAndId(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('Recognizer/trainingdata2.yml')
cv2.destroyAllWindows()
    
