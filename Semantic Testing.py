import cv2
import numpy as np
import tensorflow
import keras

from keras_applications import resnet50
import segmentation_models as sm
from keras.models import Sequential
from PIL import Image
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from itertools import product



scaler = MinMaxScaler()

# from smooth_tiled_predictions import predict_img_with_smooth_windowing


def preprocessing():
    pass


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
    return segmented_img


# changed the image file
img = cv2.imread("rs100.png", 1)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
prepeocessed = preprocess_input(img)


from keras.models import load_model


# changed the model location
model = load_model("model_epoch88.keras", compile=False)
patch_size = 128

# Number of classes
# n_classes = 31
SIZE_X = (img.shape[1]//patch_size)*patch_size      # Nearest size divisible by our patch size
SIZE_Y = (img.shape[0]//patch_size)*patch_size      # Nearest size divisible by our patch size
large_img = Image.fromarray(prepeocessed)
large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))      # Crop from top left corner
#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
large_img = np.array(large_img)

patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap
patches_img = patches_img[:, :, 0, :, :, :]
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

patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1],
                                                     patches_img.shape[2], patches_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))     # this is the final prediction with labels as the pixels

# Getting the characters from prediction image


unique = np.unique(unpatched_prediction)        # final prediction
unique_num = []
for each in unique:
    width = len(unpatched_prediction[0])
    height = len(unpatched_prediction)
    count = 0
    for i in range(0, height):
        for j in range(0, width):
            if unpatched_prediction[i][j] == each:
                count += 1
    unique_num.append(count)

for i in range(0, len(unique)):
    print(str(unique[i])+" : "+str(unique_num[i]))
# print(unique)
rgb = label_to_rgb(unpatched_prediction)
plt.imshow(rgb)
plt.axis('off')
plt.show()

# import numpy as np
#
# # Assuming 'unpatched_prediction' contains the predicted labels
#
# # Initialize an empty list to store the most repeated class label for each column
# most_repeated_labels = []
#
# # Get the number of columns in the 'unpatched_prediction' array
# num_columns = unpatched_prediction.shape[1]
#
# # Iterate through each column
# for col_idx in range(num_columns):
#     # Get the current column
#     column = unpatched_prediction[:, col_idx]
#
#     # Count the occurrences of each unique class label in the column
#     unique_labels, label_counts = np.unique(column, return_counts=True)
#
#     # Sort the unique labels by their counts in descending order
#     sorted_labels_by_count = unique_labels[np.argsort(-label_counts)]
#
#     # Get the most repeated class label for the column
#     most_repeated_label = sorted_labels_by_count[0]
#
#     # If the most repeated label is 0, choose the second most repeated label
#     if most_repeated_label == 0:
#         # If there's only one unique label, set most_repeated_label to that label
#         if len(sorted_labels_by_count) == 1:
#             most_repeated_label = sorted_labels_by_count[0]
#         else:
#             most_repeated_label = sorted_labels_by_count[1]
#
#     # Append the most repeated label to the list
#     most_repeated_labels.append(most_repeated_label)
#
# # Convert the list to a NumPy array
# most_repeated_labels_array = np.array(most_repeated_labels)
#
# print("Most repeated class labels for each column:", most_repeated_labels_array)
#
# # Example array containing most repeated class labels for each column
# # most_repeated_labels_array = np.array([1, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0])
#
# # Initialize an empty list to store new arrays without consecutive zeros
# new_arrays = []
#
# # Find indices where consecutive zeros occur
# zero_indices = np.where(np.diff(np.concatenate(([0], most_repeated_labels_array == 0, [0]))))[0]
#
# temp_zero_indices = [0]
# for zero_index in zero_indices:
#     temp_zero_indices.append(zero_index)
# temp_zero_indices.append(len(zero_indices))
# zero_indices = temp_zero_indices
# # Iterate through the zero indices and split the array accordingly
# for start, end in zip(zero_indices[:-1:2], zero_indices[1::2]):
#     new_array = most_repeated_labels_array[start:end]
#     if len(new_array) > 0:
#         new_arrays.append(new_array)
#
# # # If the last group ends with zeros, add the remaining portion to new_arrays
# # if zero_indices[-1] == len(most_repeated_labels_array) - 1 and most_repeated_labels_array[-1] == 0:
# #     new_array = most_repeated_labels_array[zero_indices[-2] + 1:]
# #     if len(new_array) > 0:
# #         new_arrays.append(new_array)
#
# # Print the new arrays without consecutive zeros
# for idx, arr in enumerate(new_arrays):
#     print(f"Array {idx + 1}:", arr)
#
# from collections import Counter
#
# # Assuming new_arrays contains the new sliced arrays without consecutive zeros
#
# # Initialize an empty list to store the most repeated values in each sliced array
# most_repeated_values = []
#
# # Iterate through each sliced array
# for arr in new_arrays:
#     # Count the occurrences of each value in the sliced array
#     value_counts = Counter(arr)
#
#     # Find the most repeated value (excluding zeros)
#     most_repeated_value = max(value_counts, key=value_counts.get)
#
#     # Append the most repeated value to the list
#     most_repeated_values.append(most_repeated_value)
#
# # Find values repeated more than 5 times in the most repeated values list
# selected_values = [value for value, count in Counter(most_repeated_values).items() if count > 5]
#
# print("Selected values repeated more than 5 times:", selected_values)



label_to_char = {0: 'rock',1: 's',2: 'sh',3: 'p',4: 'ru2',5: 'ru',6: 'm',7: 'k',8: 'li',9: 'dh',10: 'pu',11: 'th',
        12: 'u',13: 'h',14: 'le',15: 'Nhe',16: 'ch',17: 'thu',18: 'b',19: 'ri',20: 'y',21: 'shu',22: 'r',23: 'ki',
        24: 'kadhi',25: 'vi',26: 'g',27: 'thi',28: 'n',29: 'a',30: 'dhi'}


def get_char_from_label(label):
    # Assuming label_to_char dictionary is defined
    return label_to_char.get(label)


most_repeated_labels = []
# Get the number of columns in the 'unpatched_prediction' array
num_columns = unpatched_prediction.shape[1]

# Iterate through each column
for col_idx in range(num_columns):
    # Get the current column
    column = unpatched_prediction[:, col_idx]

    # Count the occurrences of each unique class label in the column
    unique_labels, label_counts = np.unique(column, return_counts=True)

    # Sort the unique labels by their counts in descending order
    sorted_labels_by_count = unique_labels[np.argsort(-label_counts)]

    # Get the most repeated class label for the column
    most_repeated_label = sorted_labels_by_count[0]

    # If the most repeated label is 0, choose the second most repeated label
    if most_repeated_label == 0:
        # If there's only one unique label, set most_repeated_label to that label
        if len(sorted_labels_by_count) == 1:
            most_repeated_label = sorted_labels_by_count[0]
        else:
            most_repeated_label = sorted_labels_by_count[1]

    # Append the most repeated label to the list
    most_repeated_labels.append(most_repeated_label)

# Convert the list to a NumPy array
most_repeated_labels_array = np.array(most_repeated_labels)

# print("Most repeated class labels for each column:", most_repeated_labels_array)


# most_repeated_labels = [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 24, 24, 24, 24, 24, 24, 24, 24, 22, 22, 22, 6, 22, 22, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 24, 24, 24, 24, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 6, 6, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 29, 29, 29, 29, 29, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
# most_repeated_labels_array = np.array(most_repeated_labels)
# Initialize an empty list to store new arrays without consecutive zeros
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

# Print the combinations
for combination in combinations:
    print(combination)