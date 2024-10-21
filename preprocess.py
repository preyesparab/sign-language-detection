# preprocess.py

import cv2
import numpy as np

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0
    return img
