import os
from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

# Load the model 
# model = tf.keras.models.load_model('wdwdwdwdwdwdwdwdwdwdwdwwdwdwd')


# Do the preprocess part when image came to backend
def preprocess():
    print("Do the preprocess here")


# testing the server
@app.route('/test')
def testing():
    print("Heritage Decode Backend Sever is working")


# prediction adn translating 
@app.route('/upload_and_translate', methods=['POST'])
def upload_and_translate():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400
        image_file = request.files['image']



