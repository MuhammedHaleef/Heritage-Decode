import os
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler

print('Work1')

# Load the model 
try:
    model = load_model('E:/2nd year/Heritage Decode/flask backend/model_epoch80.keras', compile=False)
    print('Model loaded successfully')
except Exception as e:
    print(f'Error loading the model: {e}')

patch_size = 128
print('Work3')

# Min Max Scaler
scaler = MinMaxScaler()
print('Work4')

# Do the preprocess part when image came to backend
def preprocess(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)
    # Convert image to RGB (OpenCV uses BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform any additional preprocessing here
    print("Do the preprocess here")
    return image

# Translating from the database (Lakindu's part)
def translate(processed_image):
    # Implement Lakindu's translation function here
    print("Lakindu's part here")

# prediction and translating 
def upload_and_translate(image_file):
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    processed_image = preprocess(temp_image_path)

    SIZE_X = (processed_image.shape[1]//patch_size)*patch_size
    SIZE_Y = (processed_image.shape[0]//patch_size)*patch_size
    large_img = Image.fromarray(processed_image)
    large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))
    large_img = np.array(large_img)

    patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)
    patches_img = patches_img[:,:,0,:,:,:]
    patched_prediction = []

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :, :]
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
    for i in unpatched_prediction:
        print(i)
    translated_text = " "
    segmented_image = []

    os.remove(temp_image_path)

    return {
        "Translated Text": translated_text,
        "Segmented Image": segmented_image
    }

if __name__ == '__main__':
    # For testing the function
    # Pass a mock file object, as the function expects a file-like object
    class MockFile:
        def save(self, path):
            pass

    mock_file = MockFile()
    result = upload_and_translate(mock_file)
    print(result)
