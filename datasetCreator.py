import cv2

Id="2" #Assigns unique ID to every captured face
cam = cv2.VideoCapture(0)   #starts video camera
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #creates a file containing faces data
sampleNum=0 #random number for naming each captured image
while(True):    #Continously captures faces
    ret, img = cam.read()   #Reads through camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #creates object of RGB Colors
    faces = detector.detectMultiScale(gray, 1.3, 5) #converts captured images to gray form
    for (x,y,w,h) in faces: #creates a rectangular face through captured images
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleNum=sampleNum+1
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #saves images in a local folder
        cv2.imshow('Face',img)
    if cv2.waitKey(100) & 0xFF == ord('q'): #close loop if user enters 'q'
        break
    elif sampleNum>20: #close loop if 20 samples have been collected for every faces
        break
cam.release()   #closes camera after capturing
cv2.destroyAllWindows() #closes all windows