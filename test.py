import face_recognition
import numpy as np
import cv2

imgWill = face_recognition.load_image_file('TestImages/Will Smith.jpg')
imgWill = cv2.cvtColor(imgWill,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('TestImages/Will Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
imgTest2 = face_recognition.load_image_file('TestImages/Chris Rock.jpg')
imgTest2 = cv2.cvtColor(imgTest2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgWill)[0]
encodeWill = face_recognition.face_encodings(imgWill)[0]
cv2.rectangle(imgWill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# gives top, right, bottom, and left locations for face in imgWill

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) 
# gives top, right, bottom, and left locations for face in imgTest

faceLocTest2 = face_recognition.face_locations(imgTest2)[0]
encodeTest2 = face_recognition.face_encodings(imgTest2)[0]
cv2.rectangle(imgTest2,(faceLocTest2[3],faceLocTest2[0]),(faceLocTest2[1],faceLocTest2[2]),(255,0,255),2) 
# gives top, right, bottom, and left locations for face in imgTest2

results = face_recognition.compare_faces([encodeWill], encodeTest)
faceDis = face_recognition.face_distance([encodeWill], encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
# gives us a true/false value for the comparison of the two faces and the distance between the two (which gives us a confidence level) and adds text to the image

print(results)

results2 = face_recognition.compare_faces([encodeWill], encodeTest2)
faceDis2 = face_recognition.face_distance([encodeWill], encodeTest2)
cv2.putText(imgTest2,f'{results2} {round(faceDis2[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
# gives us a true/false value for the comparison of the two faces and the distance between the two (which gives us a confidence level) and adds text to the image


cv2.imshow('Will Smith', imgWill)
cv2.imshow('Will Test', imgTest)
cv2.imshow('Chris Rock Test', imgTest2)
cv2.waitKey(0)
# prints results and shows the three images with rectangles and whether they match the reference image