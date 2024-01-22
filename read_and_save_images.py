import os.path
import random
import math
import cv2
import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator

Categories1 = ["g", "k", "m", "sh"]
base_path = "C:/archi/train2"
train = "C:/archi/train3/train"
validate = "C:/archi/train3/validate"

dataGen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05,
                             height_shift_range=0.05
                             )

source = "C:/archi/training"
destination = "C:/archi/train4"

train_path = os.path.join(destination, "train")
test_path = os.path.join(destination, "test")

os.mkdir(train_path)
os.mkdir(test_path)
generated = os.path.join(destination, "generated")
os.mkdir(generated)

def findedges(image_input):
    pass
    fft = np.fft.fft2(image_input)
    f_shift = np.fft.fftshift(fft)
    # magnitude = 20*np.log(np.abs(f_shift))

    rows = image_input.shape[0]
    coloumns = image_input.shape[1]

    center_row = rows // 2
    center_col = coloumns // 2
    f_shift[center_row - 30: center_row + 30, center_col - 30:center_col + 30] = 0

    f_inv_shift = np.fft.ifftshift(f_shift)

    image_edges = np.fft.ifft2(f_inv_shift)
    image_edges = np.abs(image_edges)

    return image_edges


def applyThreshold(image_input):
    threshold_value = 110
    image_input = cv2.resize(image_input, (80, 80))
    thresh_image = cv2.threshold(image_input, threshold_value, 255.0, cv2.THRESH_BINARY)

    return thresh_image

def get_neighbours(image_input,i, l):
    neighbours = []
    sorounding = []
    try:

        neighbours.append(image_input[i - 1][l - 1])
        neighbours.append(image_input[i - 1][l])
        neighbours.append(image_input[i - 1][l + 1])
        neighbours.append(image_input[i][l - 1])
        neighbours.append(image_input[i][l + 1])
        neighbours.append(image_input[i + 1][l - 1])
        neighbours.append(image_input[i + 1][l])
        neighbours.append(image_input[i + 1][l + 1])

        sorounding.append(image_input[i - 2][l - 2])
        sorounding.append(image_input[i - 2][l - 1])
        sorounding.append(image_input[i - 2][l])
        sorounding.append(image_input[i - 2][l - 1])
        sorounding.append(image_input[i - 2][l + 2])
        sorounding.append(image_input[i - 1][l - 2])
        sorounding.append(image_input[i][l - 2])
        sorounding.append(image_input[i + 1][l - 2])
        sorounding.append(image_input[i + 2][l - 2])
        sorounding.append(image_input[i + 2][l - 1])
        sorounding.append(image_input[i + 2][l])
        sorounding.append(image_input[i + 2][l + 1])
        sorounding.append(image_input[i + 2][l + 2])
        sorounding.append(image_input[i - 1][l + 2])
        sorounding.append(image_input[i][l + 2])
        sorounding.append(image_input[i + 1][l + 2])

    except Exception as e:
        pass
    return neighbours, sorounding


def removeSpots(image_input):

    for i in range(0, len(image_input)):
        row = image_input[i]
        for l in range(0, len(row)):
            nei, sur = get_neighbours(image_input,i, l)
            count_nei = 0
            count_ser = 0
            for r in nei:
                if r == 255:
                    count_nei += 1
            for k in sur:
                if k == 255:
                    count_ser += 1
            if count_nei < 4 and count_ser < 5:
                image_input[i][l] = 0

    return image_input


for cat in Categories1:
    folder_path = os.path.join(source, cat)
    savePath = os.path.join(generated, cat)
    os.mkdir(savePath)
    images = []
    for img in os.listdir(folder_path):
        try:
            image = np.expand_dims(cv2.imread(os.path.join(folder_path, img)), 0)
            dataGen.fit(image)
            for x, val in zip(dataGen.flow(image, save_to_dir=savePath, save_prefix=cat,
                                           save_format='png'), range(10)):
                pass

        except Exception as e:
            print(e.with_traceback())

all_images = []
nums = []
for cat in Categories1:
    folder = os.path.join(generated, cat)
    images = []
    count = 0
    for img in os.listdir(folder):
        images.append([cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE), img])
        count += 1
    nums.append(count)
    random.shuffle(images)
    all_images.append(images)

lowest = int(min(nums) * 3 / 4)

for i in range(0, len(Categories1)):
    cat = Categories1[i]
    train_p = os.path.join(train_path, cat)
    test = os.path.join(test_path, cat)
    os.mkdir(train_p)
    os.mkdir(test)
    for image in all_images[i][:lowest]:
        thresh = applyThreshold(image[0])
        cleaned = removeSpots(thresh[1])
        edges = findedges(cleaned)
        cv2.imwrite(os.path.join(train_p, image[1]), edges)

    for image in all_images[i][lowest:]:
        thresh = applyThreshold(image[0])
        cleaned = removeSpots(thresh[1])
        edges = findedges(cleaned)
        cv2.imwrite(os.path.join(test, image[1]), edges)
