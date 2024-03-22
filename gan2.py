import array
import os

import cv2
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.src.utils import image_dataset_from_directory

image_rows = 80
image_coloumns = 80
channels = 1

image_shape = (image_rows, image_coloumns, channels)


def build_generator():
    noise_shape = (80,)
    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(image_shape), activation='tanh'))
    model.add(Reshape(image_shape))

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


def build_discriminator():
    model = Sequential()

    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=image_shape)
    validity = model(img)

    return Model(img, validity)


def train(epochs, batch_size=12, save_interval=50):
    path = "C:/archi/train5"
    data_generator = ImageDataGenerator(rescale=1. / 255)
    train_generator = data_generator.flow_from_directory(
        path,
        target_size=(80, 80,1),  # Set the target size of the images
        batch_size=2,
        class_mode='binary',  # Set the class mode based on your task (binary, categorical, etc.)
        # subset='training'  # Specify 'training' to load the training set
    )

    # x_train, y_train = train_generator.next()
    # x_train = (x_train.astype(np.float32)) / 255
    # x_train = np.expand_dims(x_train, axis=0)
    # print(x_train.shape)
    # dataset = image_dataset_from_directory(path, label_mode=None, image_size=(80, 80),
    #                                        batch_size=batch_size,labels="inferred")

    image_generator = ImageDataGenerator(preprocessing_function=lambda x: (x.astype("float32") - 127.5) / 127.5)
    dataset = image_generator.flow_from_directory(
        path,
        target_size=(80, 80),
        batch_size=10,
        class_mode=None,
        shuffle=True,
    )

    x_train = []
    #
    x_train1=array.array("i")

    # for img in os.listdir(path):
    #     image = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image,(80,80))
    #     # image=np.expand_dims(image,axis=2)
    #     k=image.size
    #     x_train.append(image.astype(np.float32)/255)

        # x_train.append(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255)

    # for each in x_train:
        # x_train1.append(np.expand_dims(each, axis=2))
    # x_train=np.expand_dims(x_train,axis=0)

    print(len(x_train1))


    half_bach = int(batch_size / 2)

    for epoch in range(epochs):
        for batch in dataset:
            # idx = np.random.randint(0,len(x_train), half_bach)
            # imgs1=[]
            # for each in idx:
            #     imgs1.append(x_train[each])

        # imgs = x_train[idx]
            noise = np.random.normal(0, 1, (batch.shape[0], 80))
            fake_images = generator.predict(noise)
            # noise = np.random.normal(0, 1, (half_bach, 100))
            # gen_imgs = generator.predict(noise)
            combined = np.concatenate([batch,fake_images])
            labels = np.concatenate([np.ones((batch.shape[0], 1)), np.zeros((batch.shape[0], 1))])
            labels += 0.05 * np.random.random(labels.shape)
            v=np.ones((half_bach,1))

            d_loss_real = discriminator.train_on_batch(combined,labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_bach, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal((0, 1, (batch_size, 100)))

            valid_y = np.array([1] * batch_size)
            g_loss = combined.train_on_batch(noise, valid_y)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                save_imgs(epoch)


def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()


optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])


generator = build_generator()

generator.compile(loss='binary_crossentropy', optimizer=optimizer)

z = Input(shape=(80,))   #Our random input to the generator
img = generator(z)

discriminator.trainable = False
valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


train(epochs=100, batch_size=32, save_interval=10)

generator.save('generator_model.h5')
