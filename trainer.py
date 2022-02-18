import os
import numpy as np
from PIL import Image
import cv2

def train(name):
    path = 'datasets'
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def imagesandlabels(path):
        Imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
        samples=[]
        ids=[]
        for Imagepath in Imagepaths:
            PIL_img = Image.open(Imagepath).convert('L')
            img_numpy= np.array(PIL_img, 'uint8')
            id = int(os.path.split(Imagepath)[-1].split(".")[2])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                samples.append(img_numpy[x:x+w, y:y+h])
                ids.append(id)
        return samples, ids

    faces, ids = imagesandlabels(path)
    recognizer.train(faces,np.array(ids))

    recognizer.save("trainer/trainer_"+name+".yml")
    print("\nSaving trainer..... \nExiting")

