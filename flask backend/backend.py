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
        SIZE_X = (processed_image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
        SIZE_Y = (processed_image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
        large_img = Image.fromarray(processed_image)
        large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
        large_img = np.array(large_img)

        patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        patches_img = patches_img[:,:,0,:,:,:]
        patched_prediction = []
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :, :]

        # Use minmaxscaler instead of just dividing by 255.
                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
                    single_patch_img.shape)
                single_patch_img = np.expand_dims(single_patch_img, axis=0)
                pred = model.predict(single_patch_img)
                pred = np.argmax(pred, axis=3)
                pred = pred[0, :, :]

                patched_prediction.append(pred)

        patched_prediction = np.array(patched_prediction)
        print(patched_prediction.shape)
        print(patches_img.shape)
        patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1],
                                                     patches_img.shape[2], patches_img.shape[3]])

        unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))
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
