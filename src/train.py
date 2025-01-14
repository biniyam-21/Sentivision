"""
Sentivision - Model Training Pipeline for FER-2013
"""
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import build_emotion_model

def get_data_generators(train_dir='data/train', val_dir='data/test', batch_size=64):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )
    return train_gen, val_gen

def train_model(epochs=50, batch_size=64):
    train_gen, val_gen = get_data_generators(batch_size=batch_size)
    model = build_emotion_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    model.save_weights('model.h5')
    return history

if __name__ == '__main__':
    train_model()
