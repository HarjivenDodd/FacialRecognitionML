import face_recognition
import numpy as np
import cv2

imgWill = face_recognition.load_image_file('TestImages/Will Smith.jpg')
imgWill = cv2.cvtColor(imgWill,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('TestImages/Will Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgWill)[0]
encodeWill = face_recognition.face_encodings(imgWill)[0]
cv2.rectangle(imgWill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # gives top, right, bottom, and left locations
# gives top, right, bottom, and left locations for face in imgWill

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) 
# gives top, right, bottom, and left locations for face in imgTest

results = face_recognition.compare_faces([encodeWill], encodeTest)
faceDis = face_recognition.face_distance([encodeWill], encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)

print(results)

cv2.imshow('Will Smith', imgWill)
cv2.imshow('Will Test', imgTest)
cv2.waitkey(0)