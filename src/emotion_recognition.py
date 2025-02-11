"""
Sentivision - Real-Time Facial Emotion Recognition Engine
"""
import os
import argparse
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import build_emotion_model
from utils import EMOTION_DICT, get_asset_path, draw_hud, plot_model_history

def run_display():
    model_path = get_asset_path('model.h5')
    cascade_path = get_asset_path('haarcascade_frontalface_default.xml')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade detector not found at: {cascade_path}")

    model = build_emotion_model()
    model.load_weights(model_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped = cv2.resize(roi_gray, (48, 48))
            tensor = np.expand_dims(np.expand_dims(cropped, -1), 0) / 255.0
            
            prediction = model.predict(tensor, verbose=0)[0]
            max_idx = int(np.argmax(prediction))
            emotion_label = EMOTION_DICT.get(max_idx, "Unknown")
            confidence = prediction[max_idx] * 100.0

            draw_hud(frame, (x, y, w, h), emotion_label, confidence)

        cv2.imshow('Sentivision - Live Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_training():
    from train import train_model
    print("[INFO] Initiating model training on FER-2013 dataset...")
    history = train_model(epochs=50)
    plot_model_history(history, save_path=get_asset_path('plot.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentivision Facial Emotion Engine")
    parser.add_argument("--mode", choices=["train", "display"], default="display", help="Run 'train' or 'display'")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    else:
        run_display()
