"""
Sentivision - Utilities for plotting, path resolution, and video HUD drawing
"""
import os
import matplotlib.pyplot as plt
import cv2

EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_asset_path(filename):
    return os.path.join(get_base_dir(), filename)

def plot_model_history(model_history, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(model_history.history['accuracy']) + 1)
    
    axs[0].plot(epochs, model_history.history['accuracy'], label='train')
    axs[0].plot(epochs, model_history.history['val_accuracy'], label='val')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='best')

    axs[1].plot(epochs, model_history.history['loss'], label='train')
    axs[1].plot(epochs, model_history.history['val_loss'], label='val')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='best')

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    plt.show()

def draw_hud(frame, face_coords, emotion_label, confidence):
    (x, y, w, h) = face_coords
    cv2.rectangle(frame, (x, y - 10), (x + w, y + h + 10), (0, 255, 0), 2)
    label_text = f"{emotion_label}: {confidence:.1f}%"
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    badge_y1 = max(y - 35, 0)
    badge_y2 = max(y - 10, text_h + baseline)
    cv2.rectangle(frame, (x, badge_y1), (x + text_w + 10, badge_y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, label_text, (x + 5, badge_y2 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
