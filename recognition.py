import os
import cv2
import sys
from PIL import Image

def main(name):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer_"+name+".yml")
    CascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(CascPath) 

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480)

    minW=0.1*video_capture.get(3)
    minH=0.1*video_capture.get(4)
    flag=0
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize= (int(minW), int(minH))
        )
        for (x,y,w,h) in faces:
            
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            img= frame[y:y+h, x:x+w]
            if confidence<100:
                confid = round(100-confidence)
    
            else:
                confid = round(100-confidence)
            
            if int(confid)>50:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                cv2.putText(
                frame,
                str(name),
                (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,255,0),
                3
                )
                flag+=1
            else:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
                cv2.putText(
                frame,
                str('Unknown'),
                (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                1
                )
                flag-=1
           
        cv2.imshow('Video', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            if flag>0:
                dim= (124,124)
                resized= cv2.resize(img, dim, interpolation= cv2.INTER_AREA)
                cv2.imwrite('xtras/user.'+str(name)+'.jpg', resized)
                img1=Image.open('2.png')
                img2= Image.open('xtras/user.'+str(name)+'.jpg')
                img1copy=img1.copy()
                img1copy.paste(img2, (195,114))
                img1copy.save('xtras/end.png')
                frame = cv2.imread('xtras/end.png', 1)

                cv2.imshow('Result', frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27: # press 'ESC' to quit
                    break
            
    


    print("\nEnding the program")
    video_capture.release()
    cv2.destroyAllWindows()

    



