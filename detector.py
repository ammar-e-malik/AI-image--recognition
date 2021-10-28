import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cam=cv2.VideoCapture(0) 
rec=cv2.face.LBPHFaceRecognizer_create() 
rec.read("trainer\\trainingData.yml")   
id=0    
name="NONE"
font=cv2.FONT_HERSHEY_COMPLEX_SMALL     #sets label font
while(True):        #continously monitor camera
    ret, img = cam.read()       #reads from camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #set color to gray
    faces=faceDetect.detectMultiScale(gray, 1.3, 5) #sets dimensions of face
    for (x,y,w,h) in faces:                         #loop for face detection
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #crops image to face
        id,conf=rec.predict(gray[y:y+h,x:x+w])      #predicts face
        print(id)           #prints id of face       
        if id==1:           #sets label of face detected
            name="Mam"
        elif id==2:
            name="Noman"
     #  elif id==3:
      #      name="Wahaj"
        else:
            name="NONE"
        cv2.putText(img,name,(x,y+h),font,6,(0,0,255),4)    #sets label on detected face
    cv2.imshow("Face",img)              
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()