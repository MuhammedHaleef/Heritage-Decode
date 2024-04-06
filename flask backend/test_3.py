import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from itertools import product
from keras.models import load_model
from PIL import Image
from fuzzywuzzy import fuzz
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler
import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        database="archi"
    )

    # Create a cursor to interact with the MySQL server
    cursor = conn.cursor()
    print("Connection established successfully!")

except Exception as e:
    print("Error connecting to the database:", e)


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


def preprocess(img):
    # img_processed = cv2.convertScaleAbs(img, alpha=1, beta=1)  # Set exposure to -1
    # img_processed = cv2.convertScaleAbs(img_processed, alpha=-1, beta=80)  # Set contrast to 60
    #
    # return img_processed
    return img

    def to_letter(unpatched_prediction):
        label_to_char = {0: 'rock', 1: 's', 2: 'sh', 3: 'p', 4: 'ru2', 5: 'ru', 6: 'm', 7: 'k', 8: 'li', 9: 'dh',
                         10: 'pu', 11: 'th',
                         12: 'u', 13: 'h', 14: 'le', 15: 'Nhe', 16: 'ch', 17: 'thu', 18: 'b', 19: 'ri', 20: 'y',
                         21: 'shu', 22: 'r', 23: 'ki',
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


model = load_model("C:/archi\models\model_epoch83.keras", compile=False)
patch_size = 128
print('Work3')

# Min Max Scaler

scaler = MinMaxScaler()
print('Work4')


# Do the preprocess part when image came to backend
# def preprocess(img):
#     img_processed = cv2.convertScaleAbs(img, alpha=-1, beta=1)  # Set exposure to -1
#     img_processed = cv2.convertScaleAbs(img_processed, alpha=-1, beta=50)  # Set contrast to 60
#     return img_processed


# Translating from the database (Lakindu's part)
def translate():
    print("Lakindu's part here")


def to_letter(unpatched_prediction):
    label_to_char = {0: 'rock', 1: 's', 2: 'sh', 3: 'p', 4: 'ru2', 5: 'ru', 6: 'm', 7: 'k', 8: 'li', 9: 'dh', 10: 'pu',
                     11: 'th',
                     12: 'u', 13: 'h', 14: 'le', 15: 'Nhe', 16: 'ch', 17: 'thu', 18: 'b', 19: 'ri', 20: 'y', 21: 'shu',
                     22: 'r', 23: 'ki',
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


def check(word):
    pass


def translate_sentence(sentence, connection):
    cursor = connection.cursor()

    # Preprocess the sentence for matching
    processed_sentence = sentence.lower()

    # Look up the sentence in the translations table
    sql = "SELECT ancient_word, english_translation FROM translations"
    cursor.execute(sql)
    all_translations = cursor.fetchall()

    # Perform fuzzy matching
    best_match = max(all_translations, key=lambda x: fuzz.ratio(processed_sentence, x[0].lower()))

    # If the best match has a good similarity, return the translation
    if fuzz.ratio(processed_sentence, best_match[0].lower()) >= 80:
        return best_match[1], fuzz.ratio(processed_sentence, best_match[0].lower())
    else:
        return False,6


def translate_word(sentence, connection):
    cursor = connection.cursor()

    # Preprocess the sentence for matching
    processed_sentence = sentence.lower()

    # Look up the sentence in the translations table
    sql = "SELECT brahmi, english FROM words"
    cursor.execute(sql)
    all_translations = cursor.fetchall()

    # Perform fuzzy matching
    best_match = max(all_translations, key=lambda x: fuzz.ratio(processed_sentence, x[0].lower()))

    # If the best match has a good similarity, return the translation
    if fuzz.ratio(processed_sentence, best_match[0].lower()) >= 80:
        return best_match[1]
    else:
        return False


# testing the server


# prediction adn translating

def upload_and_translate():
    temp_image_path = "C:/archi/test/long_4.png"
    img = cv2.imread(temp_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    processed_image = preprocess(img)

    plt.imshow(processed_image)
    plt.show()
    processed_image = processed_image

    # prediction = model.predict(processed_image)
    # call the lakindu's translating function and checking with the database

    SIZE_X = (processed_image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
    SIZE_Y = (processed_image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
    large_img = Image.fromarray(processed_image)
    large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    large_img = np.array(large_img)

    patches_img = patchify(large_img, (patch_size, patch_size, 3),
                           step=patch_size)  # Step=256 for 256 patches means no overlap
    patches_img = patches_img[:, :, 0, :, :, :]
    patched_prediction = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :, :]

            # Use minmaxscaler instead of just dividing by 255.
            single_patch_img = scaler.fit_transform(
                single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
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
    rgb = label_to_rgb(unpatched_prediction)
    plt.imshow(rgb)
    plt.show()
    combinations = to_letter(unpatched_prediction)
    combinations = [["p", "ru", "m", "k", "li"], ["pu", "th"]]
    all_words = []
    for each in combinations:
        length = len(each)
        start = 0
        end = 0
        words = []
        while start < length:
            for i in range(start, length):
                word = each[start:i]
                word = ''.join(word)
                word = translate_word(word, conn)
                if word != False:
                    start = i
                    words.append(word)
                    break
                if i == length - 1 and word == False:
                    start = length
        all_words.append(words)
    all = []
    all_score = []
    for each in all_words:
        sentence = []
        for word in each[:len(each) - 1]:
            sentence.append(word)
            sentence.append("-")
        if len(each)>1:
            sentence.append(each[len(each) - 1])
        sentence = "".join(sentence)
        translated, score = translate_sentence(sentence, conn)
        if translated:
            all.append(translated)
            all_score.append(score)
    if len(all_score)==0:
        return " unable to translate"
    else:
        max = np.argmax(all_score)
        index = all_score.index(max)
        translation = all[index]
        return translation


upload_and_translate()
