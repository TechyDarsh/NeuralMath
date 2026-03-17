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
    img_h, img_w = thresh.shape
    min_w = max(8, int(img_w * 0.005))
    min_h = max(8, int(img_h * 0.005))
    
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_w and h > min_h:  # Dynamic filtering for size
            # Ignore extremely large boxes (likely the whole screen or noise)
            if w < img_w * 0.8 and h < img_h * 0.8:
                rects.append([x, y, w, h])
            
    # Merge overlapping or touching rectangles
    def is_overlapping(r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        
        intersect_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        intersect_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        
        # 1. One box is almost entirely inside another (e.g., noise inside a character)
        if intersect_x > min(w1, w2) * 0.8 and intersect_y > min(h1, h2) * 0.8:
            return True
            
        # 2. Vertically separate but horizontally aligned (like '=' or 'i')
        if intersect_x > min(w1, w2) * 0.5:
            dist_y = max(y1, y2) - min(y1+h1, y2+h2)
            if dist_y < max(h1, h2) * 1.5: # separated by small vertical gap
                return True
                
        return False
        
    merged = True
    while merged:
        merged = False
        new_rects = []
        while len(rects) > 0:
            r = rects.pop(0)
            has_merged = False
            for i, other in enumerate(new_rects):
                if is_overlapping(r, other):
                    x1, y1, w1, h1 = r
                    x2, y2, w2, h2 = other
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1+w1, x2+w2) - nx
                    nh = max(y1+h1, y2+h2) - ny
                    new_rects[i] = [nx, ny, nw, nh]
                    has_merged = True
                    merged = True
                    break
            if not has_merged:
                new_rects.append(r)
        rects = new_rects

    valid_contours = rects
            
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
