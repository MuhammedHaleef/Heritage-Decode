import os
from numpy.random import randint
from numpy.random import randn

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
from keras.layers import Conv2D,LeakyReLU,Flatten,Dropout,Dense,Reshape,Conv2DTranspose
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.callbacks import TensorBoard
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
from numpy import ones
from keras.models import load_model


def define_discriminator(in_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))

    optimizer = Adam(learning_rate=0.00002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acuracy'])
    return model

def define_generator(latent_dim):
    model =Sequential()
    num_nodes = 128*8*8

    model.add(Dense(num_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8,8,128)))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3,(8,8),activation='tanh',padding='same'))
    return model


def define_gan(generator,discriminator):
    discriminator.trainable =False

    model =Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)

    return model

def train_model(g_model,d_model,gan_model,dataset,latent_dim,n_epochs=100,n_batch=128):
    bat_per_epoch = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)

    for i in range(n_epochs):
        for j in range(bat_per_epoch):

            x_real,y_real = generate_real_samples()
            d_loss1,_ =d_model.train_on_batch(x_real,y_real)
            x_fake,y_fake = generate_fake_samples()
            d_loss2,_ = d_model.train_on_batch(x_fake,y_fake)

            x_gan = generate_latent_points(latent_dim,n_batch)
            y_gan = ones((n_batch,1))
            g_loss = gan_model.train_on_batch(x_gan,y_gan)

            g_model.save("gan.keras")

def generate_real_samples(dataset,n_samples):
    ix = randint(0,dataset.shape[0],n_samples)
    x= dataset[ix]
    y=ones((n_samples,1))
    return x,y

def generate_fake_samples():
    pass

def generate_latent_points(latent_dim,n_samples):
    x_input = randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input
