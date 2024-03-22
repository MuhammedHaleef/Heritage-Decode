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



# Translating from the database (Lakindu's part)
def translate():
    print("Lakindu's part here")


# testing the server
@app.route('/test')
def testing():
    return("Heritage Decode Backend Sever is working")


# prediction adn translating 
@app.route('/upload_and_translate', methods=['POST'])
def upload_and_translate():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400
        image_file = request.files['image']
        
        
    
        temp_image_path = 'temp_image.jpg'
        image_file.save(temp_image_path)

        processed_image = preprocess(temp_image_path)
        
    
        #prediction = model.predict(processed_image)
        # call the lakindu's translating function and checking with the database

        translated_text = " "
        segmented_image = []

        # After that show the original image, segmanted image and the translated text here

        os.remove(temp_image_path)

        return jsonify({
                   "Translated Text":translated_text,
                    "Segmented Image":segmented_image
                       })

        






if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)  