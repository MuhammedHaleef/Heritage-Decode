import os

import cv2
import numpy as np
import keras
from keras import Model
import tensorflow as tf
# from keras.applications.mobilenet_v3 import preprocess_input
# from keras.applications.mobilenet_v3 import
# from keras_applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from keras.preprocessing.image import
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.callbacks import TensorBoard
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

train_path = "C:/archi/train4/train"
test_path = "C:/archi/train4/test"


def applyThreshhold(image):
    threshold_value = 128
    # image[image <= threshold_value] = 0
    # image[image > threshold_value] = 255
    # return image
    #     threshHold_value = 128
    image = cv2.resize(image, (80, 80))
    # image.resize(60, 60)
    thresh_image = cv2.threshold(image, threshold_value, 255.0,cv2.THRESH_BINARY)
    #     # thresh_image = np.array(thresh_image)
    #     # thresh_image = cv2.resize(thresh_image,(60,60))
    #     # thresh_image.resize(60, 60)
    return thresh_image
    #     image = cv2.resize(image,(60,60))
    # image = cv2.resize(image, (60, 60))
    # image_1 = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # return image_1


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(train_path,
                                                                                           target_size=(80, 80),
                                                                                           batch_size=10
                                                                                           )
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(test_path,
                                                                                          target_size=(80, 80),
                                                                                          batch_size=10)


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(80, 80, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

preds = Dense(4, activation='softmax')(x)

model_1 = Model(inputs=base_model.input, outputs=preds)

for layers in model_1.layers[:-5]:
    layers.trainable = False

num_epochs = 5
optimizer = Adam(learning_rate=0.002)
model_1.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model_1.fit(train_gen, validation_data=test_gen, epochs=num_epochs)
model_1.save("C:/archi/models")
# model_1.summary()