from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from pygame import mixer

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
# src = cv2.imread("allu.jpg") 
# Above line is used to read face from local directory.
cam = cv2.VideoCapture(0)
ret,src = cam.read()

mixer.init()

# Add music files based on emotions in the root directory.
def playMusic(index):
    if(index==0):
        mixer.music.load("song.mp3") # Angry emotion based music
    elif(index==1):
        mixer.music.load("song.mp3") # Happy emotion based music
    elif(index==2):
        mixer.music.load("song.mp3") # Neutral emotion based music
    elif(index==3):
        mixer.music.load("song.mp3") # Sad emotion based music
    elif(index==4):
        mixer.music.load("song.mp3") # Surprise emotion based music
    mixer.music.play()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.3, 4)
while len(faces)==0:
    ret,src = cam.read()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 4)

cam.release()


for (x, y, w, h) in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = classifier.predict(roi)[0]
        print("\nPredictions = ", preds)
        label = class_labels[preds.argmax()]
        print("\nAccurate Index = ", preds.argmax())
        print("\nEmotion = ", label)
        label_position = (x, y)
        cv2.putText(src, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(src, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        print("\n\n")

cv2.imshow("Output",src)
playMusic(preds.argmax())
cv2.waitKey(0)