import os
from collections import deque
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Define constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]  # Update with your classes

# Load the pre-trained model
model = load_model('my_model.h5')  # Update with your model path

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255
    return normalized_frame

def frames_extraction(video_path):
    video_reader = cv2.VideoCapture(video_path)
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            frames_array = np.array(frames_queue)
            prediction = model.predict(np.expand_dims(frames_array, axis=0))[0]
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASSES_LIST[predicted_class_index]
    video_reader.release()
    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

import logging

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        video_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(video_path)
        predicted_class = frames_extraction(video_path)
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True)
