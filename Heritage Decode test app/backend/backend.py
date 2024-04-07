import os
from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from collections import Counter
from itertools import product


app = Flask(__name__)

#Testing the sever
@app.route('/test')
def test():
    return ('Heritage Decode is working')

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



    img_processed = cv2.convertScaleAbs(image, alpha=-1, beta=1)  # Set exposure to -1
    img_processed = cv2.convertScaleAbs(img_processed, alpha=-1, beta=50)  # Set contrast to 60
    return img_processed

# Translating from the database (Lakindu's part)
def translate(processed_image):
    # Implement Lakindu's translation function here
    print("Lakindu's part here")

# Converting labels to RGB values   
def label_to_rgb(predicted_image):

    s = '#ff000f'.lstrip('#')
    s = np.array(tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)))

    sh = '#650006'.lstrip('#')
    sh = np.array(tuple(int(sh[i:i + 2], 16) for i in (0, 2, 4)))

    p = '#0f00ff'.lstrip('#')
    p = np.array(tuple(int(p[i:i + 2], 16) for i in (0, 2, 4)))

    ru2 = '#6713ec'.lstrip('#')
    ru2 = np.array(tuple(int(ru2[i:i + 2], 16) for i in (0, 2, 4)))

    ru = '#070348'.lstrip('#')
    ru = np.array(tuple(int(ru[i:i + 2], 16) for i in (0, 2, 4)))

    m = '#5e5c80'.lstrip('#')
    m = np.array(tuple(int(m[i:i + 2], 16) for i in (0, 2, 4)))

    k = '#8c4747'.lstrip('#')
    k = np.array(tuple(int(k[i:i + 2], 16) for i in (0, 2, 4)))

    li = '#ff03a1'.lstrip('#')
    li = np.array(tuple(int(li[i:i + 2], 16) for i in (0, 2, 4)))

    dh = '#0ffe00'.lstrip('#')
    dh = np.array(tuple(int(dh[i:i + 2], 16) for i in (0, 2, 4)))

    pu = '#7f7ce7'.lstrip('#')
    pu = np.array(tuple(int(pu[i:i + 2], 16) for i in (0, 2, 4)))

    th = '#ff9595'.lstrip('#')
    th = np.array(tuple(int(th[i:i + 2], 16) for i in (0, 2, 4)))

    u = '#681d67'.lstrip('#')
    u = np.array(tuple(int(u[i:i + 2], 16) for i in (0, 2, 4)))

    h = '#f7e500'.lstrip('#')
    h = np.array(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    le = '#00f4ff'.lstrip('#')
    le = np.array(tuple(int(le[i:i + 2], 16) for i in (0, 2, 4)))

    Nhe = '#6e6c93'.lstrip('#')
    Nhe = np.array(tuple(int(Nhe[i:i + 2], 16) for i in (0, 2, 4)))

    ch = '#3b8b9c'.lstrip('#')
    ch = np.array(tuple(int(ch[i:i + 2], 16) for i in (0, 2, 4)))

    thu = '#14593b'.lstrip('#')
    thu = np.array(tuple(int(thu[i:i + 2], 16) for i in (0, 2, 4)))

    b = '#fe8100'.lstrip('#')
    b = np.array(tuple(int(b[i:i + 2], 16) for i in (0, 2, 4)))

    ri = '#4fa681'.lstrip('#')
    ri = np.array(tuple(int(ri[i:i + 2], 16) for i in (0, 2, 4)))

    y = '#b7e29f'.lstrip('#')
    y = np.array(tuple(int(y[i:i + 2], 16) for i in (0, 2, 4)))

    shu = '#dd4d4d'.lstrip('#')
    shu = np.array(tuple(int(shu[i:i + 2], 16) for i in (0, 2, 4)))

    r = '#aa6eff'.lstrip('#')
    r = np.array(tuple(int(r[i:i + 2], 16) for i in (0, 2, 4)))

    ki = '#b81dba'.lstrip('#')
    ki = np.array(tuple(int(ki[i:i + 2], 16) for i in (0, 2, 4)))

    kadhi = '#835812'.lstrip('#')
    kadhi = np.array(tuple(int(kadhi[i:i + 2], 16) for i in (0, 2, 4)))

    vi = '#005c86'.lstrip('#')
    vi = np.array(tuple(int(vi[i:i + 2], 16) for i in (0, 2, 4)))

    g = '#517885'.lstrip('#')
    g = np.array(tuple(int(g[i:i + 2], 16) for i in (0, 2, 4)))

    thi = '#898989'.lstrip('#')
    thi = np.array(tuple(int(thi[i:i + 2], 16) for i in (0, 2, 4)))

    n = '#6d0000'.lstrip('#')
    n = np.array(tuple(int(n[i:i + 2], 16) for i in (0, 2, 4)))

    a = '#718693'.lstrip('#')
    a = np.array(tuple(int(a[i:i + 2], 16) for i in (0, 2, 4)))

    dhi = '#05ff92'.lstrip('#')
    dhi = np.array(tuple(int(dhi[i:i + 2], 16) for i in (0, 2, 4)))

    rock = '#3b4139'.lstrip('#')
    rock = np.array(tuple(int(rock[i:i + 2], 16) for i in (0, 2, 4)))

    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))

    segmented_img[(predicted_image == 0)] = rock
    segmented_img[(predicted_image == 1)] = s
    segmented_img[(predicted_image == 2)] = sh
    segmented_img[(predicted_image == 3)] = p
    segmented_img[(predicted_image == 4)] = ru2
    segmented_img[(predicted_image == 5)] = ru
    segmented_img[(predicted_image == 6)] = m
    segmented_img[(predicted_image == 7)] = k
    segmented_img[(predicted_image == 8)] = li
    segmented_img[(predicted_image == 9)] = dh
    segmented_img[(predicted_image == 10)] = pu
    segmented_img[(predicted_image == 11)] = th
    segmented_img[(predicted_image == 12)] = u
    segmented_img[(predicted_image == 13)] = h
    segmented_img[(predicted_image == 14)] = le
    segmented_img[(predicted_image == 15)] = Nhe
    segmented_img[(predicted_image == 16)] = ch
    segmented_img[(predicted_image == 17)] = thu
    segmented_img[(predicted_image == 18)] = b
    segmented_img[(predicted_image == 19)] = ri
    segmented_img[(predicted_image == 20)] = y
    segmented_img[(predicted_image == 21)] = shu
    segmented_img[(predicted_image == 22)] = r
    segmented_img[(predicted_image == 23)] = ki
    segmented_img[(predicted_image == 24)] = kadhi
    segmented_img[(predicted_image == 25)] = vi
    segmented_img[(predicted_image == 26)] = g
    segmented_img[(predicted_image == 27)] = thi
    segmented_img[(predicted_image == 28)] = n
    segmented_img[(predicted_image == 29)] = a
    segmented_img[(predicted_image == 30)] = dhi

    segmented_img = segmented_img.astype(np.uint8)
    return (segmented_img)

# Extracting Letters from the predicted image
def to_letter(unpatched_prediction):
        label_to_char = {0: 'rock', 1: 's', 2: 'sh', 3: 'p', 4: 'ru2', 5: 'ru', 6: 'm', 7: 'k', 8: 'li', 9: 'dh', 10: 'pu', 11: 'th',
                         12: 'u', 13: 'h', 14: 'le', 15: 'Nhe', 16: 'ch', 17: 'thu', 18: 'b', 19: 'ri', 20: 'y', 21: 'shu', 22: 'r', 23: 'ki',
                         24: 'kadhi', 25: 'vi', 26: 'g', 27: 'thi', 28: 'n', 29: 'a', 30: 'dhi'}

        def get_char_from_label(label):
            return label_to_char.get(label)

        most_repeated_labels = []
        num_columns = unpatched_prediction.shape[1]

        for col_idx in range(num_columns):
            column = unpatched_prediction[:, col_idx]
            # Count the occurrences of each unique class label in the column
            unique_labels, label_counts = np.unique(column, return_counts=True)
            # Sort the unique labels by their counts in descending order
            sorted_labels_by_count = unique_labels[np.argsort(-label_counts)]
            most_repeated_label = sorted_labels_by_count[0]

            # If the most repeated label is 0(Rock), choose the second most repeated label
            if most_repeated_label == 0:
                # If there's only one unique label, set most_repeated_label to that label
                if len(sorted_labels_by_count) == 1:
                    most_repeated_label = sorted_labels_by_count[0]
                else:
                    most_repeated_label = sorted_labels_by_count[1]

            most_repeated_labels.append(most_repeated_label)

            most_repeated_labels_array = np.array(most_repeated_labels)
            new_arrays = []

            # Find indices where consecutive zeros occur
            zero_indices = np.where(np.diff(np.concatenate(([0], most_repeated_labels_array == 0, [0]))))[0]

            temp_zero_indices = [0]
            for zero_index in zero_indices:
                temp_zero_indices.append(zero_index)
            temp_zero_indices.append(len(most_repeated_labels))
            zero_indices = temp_zero_indices
            # Iterate through the zero indices and split the array accordingly
            for start, end in zip(zero_indices[:-1:2], zero_indices[1::2]):
                new_array = most_repeated_labels_array[start:end]
                if len(new_array) > 0:
                    new_arrays.append(new_array)

            selected_values = []

            # Step 1: Loop through each sliced array
            for arr in new_arrays:
                # Step 2: Count the occurrences of each value in the sliced array
                value_counts = Counter(arr)
                temp_arr = []
                temp_arr.extend([value for value, count in value_counts.items() if count > 10])
                if len(temp_arr) > 0:
                    selected_values.append(temp_arr)

            def convert_indices_to_chars(input_list):
                # Initialize an empty list to store the converted characters
                output_list = []

                # Iterate through each sublist in the input list
                for sublist in input_list:
                    # Map each index in the sublist to its corresponding character
                    chars = [label_to_char.get(index, 'Unknown') for index in sublist]
                    # Append the list of characters to the output list
                    output_list.append(chars)

                return output_list

            # Convert indices to characters
            output_list = convert_indices_to_chars(selected_values)

            # Use itertools.product() to generate all combinations
            combinations = list(product(*output_list))
        return combinations
# prediction and translating
@app.route('/upload_and_predict', methods=['POST']) 
def upload_and_predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found'}), 400   
        image_file = request.files['image'] 
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
        with_RGB = label_to_rgb(unpatched_prediction)
        plt.imshow(with_RGB)
        plt.show()
        letters = to_letter(unpatched_prediction)
        letter1= letters[0]
        letter_string=''
        for each in letter1:
            letter_string = letter_string+each
        #plt.imshow(with_RGB)
        #plt.show()
        #print(segmented_image)
        for i in unpatched_prediction:
            print(i)
        translated_text = " "
        segmented_image = []
    
        print(letters)
    
        os.remove(temp_image_path)
        return jsonify({'predicted_class': letter_string})

        #return {
        #    "Translated Text": translated_text,
        #    "Segmented Image": unpatched_prediction
        #}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # Pass a mock file object, as the function expects a file-like object