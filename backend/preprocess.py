import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Load image, convert to grayscale, apply Gaussian blur,
    apply adaptive thresholding, and apply morphological operations
    to remove noise and enhance digits.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read. Please check the path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to remove general noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Adaptive Thresholding (invert it so text is white and background is black)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return image, thresh
