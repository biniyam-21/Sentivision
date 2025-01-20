"""
Sentivision - Utilities for plotting and path resolution
"""
import os
import matplotlib.pyplot as plt

EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

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
