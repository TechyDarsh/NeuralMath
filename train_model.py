import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2

# Classes: '0'-'9', '+', '-', '*', '/', '=', 'x', 'y'
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
           '+', '-', '*', '/', '=', 'x', 'y']

def generate_synthetic_data(num_samples_per_class=2000):
    """
    Generate a simple synthetic dataset using OpenCV putText.
    """
    X = []
    y = []
    
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, 
             cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, 
             cv2.FONT_HERSHEY_TRIPLEX]
             
    for class_idx, char in enumerate(CLASSES):
        for i in range(num_samples_per_class):
            img = np.zeros((28, 28), dtype=np.uint8)
            
            font = fonts[np.random.randint(0, len(fonts))]
            font_scale = np.random.uniform(0.6, 1.2)
            thickness = np.random.randint(1, 3)
            
            # Positioning
            tx = np.random.randint(-4, 5)
            ty = np.random.randint(-4, 5)
            
            (text_width, text_height), baseline = cv2.getTextSize(char, font, font_scale, thickness)
            x = (28 - text_width) // 2 + tx
            y_pos = (28 + text_height) // 2 + ty
            
            cv2.putText(img, char, (x, y_pos), font, font_scale, 255, thickness)
            
            # Adding noise
            noise = np.random.randint(0, 50, (28, 28), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            X.append(img)
            y.append(class_idx)
            
    X = np.array(X).astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    return X[indices], y[indices]

def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(num_samples_per_class=1000)
    print(f"Data shape: {X.shape}")
    
    model = build_model()
    
    print("Training model...")
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'cnn_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
