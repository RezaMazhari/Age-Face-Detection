import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load the model
with open('model.json', 'r') as json_file:
    model_json = json_file.read()

# Recreate the model
age_model = model_from_json(model_json)

# Load the weights
age_model.load_weights('model_weights.weights.h5')

# age categories
AGE_BUCKETS = ["Middle", "Young", "Old"]

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def preprocess_face(face):
    face_resized = cv2.resize(face, (128, 128))  
    face_array = np.array(face_resized) / 255.0  
    face_array = np.expand_dims(face_array, axis=0)  
    return face_array

def predict_age(face):
    face_array = preprocess_face(face)
    age_prediction = age_model.predict(face_array)
    age = AGE_BUCKETS[np.argmax(age_prediction)]  
    return age

def start_detection():
    global running
    running = True
    run_detection()

def stop_detection():
    global running
    running = False

def run_detection():
    if not running:
        return
    
    ret, frame = cap.read()
    if ret:
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            age = predict_age(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        #frame 
        frame = cv2.resize(frame, (640, 480))  
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        panel.img_tk = img_tk  
        panel.configure(image=img_tk)

    panel.after(10, run_detection)

#  GUI
root = tk.Tk()
root.title("Age Detection App")
root.geometry("700x550")  

# frame for the video
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# panel to display video
panel = tk.Label(frame)
panel.pack()

# label for title
title_label = tk.Label(root, text="Age Detection Application", font=("Helvetica", 20), fg="blue")
title_label.pack(pady=(10, 0))

# Start and Stop buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="Start Detection", command=start_detection, bg="green", fg="white", width=15)
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(button_frame, text="Stop Detection", command=stop_detection, bg="red", fg="white", width=15)
stop_button.pack(side=tk.LEFT, padx=10)

# Start video capture
cap = cv2.VideoCapture(0)
running = False

# Tkinter main loop
root.mainloop()

# Release video capture
cap.release()
cv2.destroyAllWindows()
