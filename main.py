# main.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from utils import within_operating_hours

# Load the trained model
model = load_model('models/sign_language_model.h5')

# Label names (adjust as per your dataset)
labels = ['Sign1', 'Sign2', 'Sign3', 'Sign4', 'Sign5']  # Adjust based on your dataset

# Predict image function
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return labels[np.argmax(prediction)]

# Function to upload and display image
def upload_image():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
    result = predict_image(file_path)
    result_label.config(text="Prediction: " + result)

# Function for real-time video detection
def start_video_detection():
    if not within_operating_hours():
        print("The model is operational only between 6 PM and 10 PM.")
        return

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (64, 64))
            img = np.expand_dims(img, axis=0) / 255.0
            prediction = model.predict(img)
            predicted_sign = labels[np.argmax(prediction)]
            
            cv2.putText(frame, predicted_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

# Set up GUI
root = tk.Tk()
root.title("Sign Language Detection")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

video_btn = tk.Button(root, text="Start Real-Time Detection", command=start_video_detection)
video_btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack()

root.mainloop()
