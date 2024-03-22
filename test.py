import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from skimage.feature import canny
import os

test_path ="C:/archi/train4/test"
def test(model):
    Categories1 = ["g", "y", "m", "sh"]
    # test_model = load_model("C:/archi/models/model1.keras")

    count = 0
    count1 = 0
    for cat in Categories1:
        path = os.path.join(test_path, cat)

        for img in os.listdir(path):
            count += 1
            image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            ed = cv2.resize(image, (80, 80))
            ed1 = np.expand_dims(ed, axis=-1)
            ed2 = np.repeat(ed1, 3, axis=-1)
            prediction = model.predict(np.expand_dims(ed2, axis=0))
            max = np.max(prediction[0])
            k = np.where(prediction[0] == max)

            predict = Categories1[k[0][0]]
            print("actual: ", cat, "  prediction:", predict)
            if cat == predict:
                count1 += 1

    print(count1 / count)
