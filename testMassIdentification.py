import face_recognition
import numpy as np
from datetime import datetime
import cv2
import os

path = 'Images'
images = []     # list of all images in 'Images' folder
className = []    # list of class names corresponding to images[]
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])

# returns a list of RGB encodings of the images in the images list/folder
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encodings Complete')

# records the name of the person in the image and the time the person was identified
def recordName(name):
    with open('names.csv','r+') as f:
        dataList = f.readlines()
        names =[]
        for line in dataList:
            entry = line.split(',')
            names.append(entry[0])
            if name not in names:
                now = datetime.now()
                dt_string = now.strftime("%H:%M:%S")
                f.writelines(f'\n{name},{dt_string}')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    imgt = cv2.resize(img, (0, 0), fx=0.25, fy=0.25) # reduces the size of images from the webcam to improve processing speed
    imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB) # converts to RGB

    facesCurFrame = face_recognition.face_locations(imgt)
    encodesCurFrame = face_recognition.face_encodings(imgt, facesCurFrame)
    # locates face in images and uses this location to encode the face in the image

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    # compares the face in the image to the faces in the images list for current frame
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex]< 0.50:
            name = className[matchIndex].upper()
            recordName(name)
        else: 
            name = 'Unidentified'
        y1,x2,y2,x1=faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 # multiplications are used to adjust the rectangles to the appropriate size of original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    # determines the closest match and displays the name of the person on the image, as well as rectangles around the face and the name
    # also rejects the face if the confidence level is less than 0.5 and displays 'Unidentified'

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    # shows the webcam feed and the name of the person identified, or 'Unidentified' if the confidence level is less than 0.5
