import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

# Map classes to characters
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
           '+', '-', '*', '/', '=', 'x', 'y']

model = None

def load_recognition_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        model = tf.keras.models.load_model(model_path)
    return model

def recognize_symbols(symbols):
    """
    Use CNN model to recognize the list of symbol images.
    """
    load_recognition_model()
    
    recognized = []
    for symbol in symbols:
        img = symbol['image']
        # Reshape to batch format
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        char = CLASSES[class_idx]
        
        symbol_info = {
            'char': char,
            'confidence': confidence,
            'box': symbol['box']
        }
        recognized.append(symbol_info)
        
    return recognized
