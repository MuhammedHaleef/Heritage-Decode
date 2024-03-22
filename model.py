import os

import cv2
import numpy as np
import keras
from keras import Model
from keras.models import Sequential
import tensorflow as tf
# from keras.applications.mobilenet_v3 import preprocess_input
# from keras.applications.mobilenet_v3 import
# from keras_applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from keras.preprocessing.image import
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D,Convolution2D,Conv2D,MaxPooling2D,Flatten,Dropout,Lambda
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.callbacks import TensorBoard
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

from keras.models import load_model

train_path = "C:/archi/train4/train"
validation_path = "C:/archi/train4/validation"

test_path ="C:/archi/train4/test"




train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(train_path,
                                                                                           target_size=(80, 80),
                                                                                           batch_size=10
                                                                                           )

validation_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(validation_path,
                                                                                          target_size=(80, 80),
                                                                                          batch_size=10)


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(80, 80, 3))
x = base_model.output
# x=GlobalAveragePooling2D()(x)
# x= Conv2D(64,strides=(1,1),input_shape=train_gen.image_shape,kernel_size=(4,4),activation='relu')(x)
# x=MaxPooling2D(pool_size=(2, 2))(x)
# x=Conv2D(64, (3, 3),activation='relu')(x)
# x=MaxPooling2D(pool_size=(2, 2))(x)
# x=Flatten()(x)
# x=Dense(64)(x)
# x=Dense(4,activation='sigmoid')
#
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(3600, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(8, activation='relu')(x)


# model.add(Conv2D(64,strides=(1,1),input_shape=train_gen.image_shape,kernel_size=(4,4),activation='relu'))
# # model.add(Activation(relu))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation(relu))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Dense(4))
# model.add(Activation('sigmoid'))
#
# model.compile(optimizer=Adam(learning_rate=0.0005,beta_1=0.1), loss=CategoricalCrossentropy(), metrics=['accuracy'])
#
# model.fit(train_gen, validation_data=validation_gen, epochs=30)

preds = Dense(4, activation='softmax')(x)

transfer_model = Model(inputs=base_model.input, outputs=preds)

# for layers in transfer_model.layers[:-5]:
#     layers.trainable = False

cls=train_gen.class_indices
print(cls)
#
num_epochs = 4
optimizer = Adam(learning_rate=0.0005   ,beta_1=0.5)
transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
transfer_model.fit(train_gen, validation_data=validation_gen, epochs=num_epochs)
transfer_model.save("C:/archi/models/model1.keras")
# transfer_model.summary()

Categories1 = ["g", "y", "m", "sh"]
test_model = load_model("C:/archi/models/model1.keras")
count=0
count1=0
for cat in Categories1:
    path = os.path.join(test_path,cat)

    for img in os.listdir(path):
        count+=1
        image = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        ed = cv2.resize(image, (80, 80))
        ed1 = np.expand_dims(ed, axis=-1)
        ed2 = np.repeat(ed1, 3, axis=-1)
        prediction = test_model.predict(np.expand_dims(ed2,axis=0))
        max = np.max(prediction[0])
        k=np.where(prediction[0]==max)

        predict = Categories1[k[0][0]]
        print("actual: ",cat,"  prediction:", predict)
        if cat==predict:
            count1+=1

print(count1/count)

