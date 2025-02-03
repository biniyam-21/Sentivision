import numpy as np
import argparse
import cv2
import os
from model import build_emotion_model

ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display", default="display")
mode = ap.parse_args().mode

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

if mode == "display":
    model = build_emotion_model()
    model.load_weights('model.h5')
    
    # Load cascade once outside the loop to optimize FPS
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)[0]
            maxindex = int(np.argmax(prediction))
            confidence = prediction[maxindex] * 100
            text = f"{emotion_dict[maxindex]}: {confidence:.1f}%"
            cv2.putText(frame, text, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
