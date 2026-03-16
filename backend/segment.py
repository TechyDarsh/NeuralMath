import cv2
import numpy as np

def segment_image(thresh):
    """
    Find contours of symbols in the thresholded image,
    sort them left-to-right, and extract 28x28 normalized images for each symbol.
    """
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to remove small noise
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:  # Basic filtering for size
            valid_contours.append((x, y, w, h))
            
    # Sort contours from left to right based on x-coordinate
    valid_contours.sort(key=lambda b: b[0])
    
    symbols = []
    for (x, y, w, h) in valid_contours:
        # Extract the region of interest
        MathRoi = thresh[y:y+h, x:x+w]
        
        # Add padding to ROI to make it square and center the symbol
        max_dim = max(w, h)
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        
        # Add a consistent 4-pixel border around the square
        padded_roi = cv2.copyMakeBorder(MathRoi, pad_y + 4, max_dim - h - pad_y + 4, 
                                        pad_x + 4, max_dim - w - pad_x + 4, 
                                        cv2.BORDER_CONSTANT, value=0)
        
        # Resize to 28x28
        resized = cv2.resize(padded_roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0-1
        normalized = resized.astype('float32') / 255.0
        # Expand dims for CNN input (1, 28, 28, 1) or (28, 28, 1) depending on model,
        # Keras predict needs (batch, 28, 28, 1) later
        normalized = np.expand_dims(normalized, axis=-1)
        
        symbols.append({
            'image': normalized,
            'box': (x, y, w, h)
        })
        
    return symbols
