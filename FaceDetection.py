import os
import cv2
import sys

def data(name):
    faceCascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480)
    count = 0
    while (True):
        ret, frame = video_capture.read()
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= 1.1,
            minNeighbors= 5,
            minSize= (30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            count+=1
            cv2.imwrite("datasets/user."+ str(name)+ "."+ str(count)+ ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Video', frame)

        if cv2.waitKey(30) and 0xFF == ord('q'):
            break
        elif count==30:
            break
    print("\nSaving the data set....... \nEnd")
    video_capture.release()
    cv2.destroyAllWindows()
    return count



