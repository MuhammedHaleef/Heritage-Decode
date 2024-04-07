import os
import cv2
import datetime as datetime

import epochs
import keras.callbacks
import numpy as np
import saver

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from keras.metrics import MeanIoU
from keras.preprocessing.image import ImageDataGenerator

seed = 24

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

root_directory = "C:/Users\MuhammedHaleef\OneDrive\Documents\AI & DS/2nd Year\CM2603 DSGP\Final Project\Labelling Dataset\Model_training"

patch_size = 128


image_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'imgs':
        images = os.listdir(path)
        for i, image_name in enumerate(images):
            if image_name.endswith(".png"):

                image = cv2.imread(path + "/" + image_name, 1)
                SIZE_X = (image.shape[1] // patch_size) * patch_size
                SIZE_Y = (image.shape[0] // patch_size) * patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, SIZE_X, SIZE_Y))

                image = np.array(image)

                # Extract patches from each image
                print("Now patchifying image:", path + "/" + image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3),
                                       step=patch_size)

                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]


                        single_patch_img = scaler.fit_transform(
                            single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)


                        single_patch_img = single_patch_img[
                            0]
                        image_dataset.append(single_patch_img)


mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':
        masks = os.listdir(path)
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".png"):

                mask = cv2.imread(path + "/" + mask_name,
                                  1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1] // patch_size) * patch_size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                mask = np.array(mask)

                # Extract patches from each image
                print("Now patchifying mask:", path + "/" + mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3),
                                        step=patch_size)

                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]

                        single_patch_mask = single_patch_mask[
                            0]
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# Sanity check, view few mages
import random
import numpy as np

image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

###########################################################################


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

m2 = '#8c89ce'.lstrip('#')
m2 = np.array(tuple(int(m2[i:i + 2], 16) for i in (0, 2, 4)))

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

# ch = '#3b8b9c'.lstrip('#')
# ch = np.array(tuple(int(ch[i:i + 2], 16) for i in (0, 2, 4)))

# thu = '#14593b'.lstrip('#')
# thu = np.array(tuple(int(thu[i:i + 2], 16) for i in (0, 2, 4)))

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

# vi = '#005c86'.lstrip('#')
# vi = np.array(tuple(int(vi[i:i + 2], 16) for i in (0, 2, 4)))

g = '#517885'.lstrip('#')
g = np.array(tuple(int(g[i:i + 2], 16) for i in (0, 2, 4)))

# thi = '#898989'.lstrip('#')
# thi = np.array(tuple(int(thi[i:i + 2], 16) for i in (0, 2, 4)))

n = '#6d0000'.lstrip('#')
n = np.array(tuple(int(n[i:i + 2], 16) for i in (0, 2, 4)))

a = '#718693'.lstrip('#')
a = np.array(tuple(int(a[i:i + 2], 16) for i in (0, 2, 4)))

# dhi = '#05ff92'.lstrip('#')
# dhi = np.array(tuple(int(dhi[i:i + 2], 16) for i in (0, 2, 4)))

rock = '#3b4139'.lstrip("#")
rock = np.array(tuple(int(rock[i:i + 2], 16) for i in (0, 2, 4)))

label = single_patch_mask



def rgb_to_2D_label(label):

    label_seg = np.zeros(label.shape, dtype=np.uint8)

    label_seg = label_seg[:, :, 0]
    label_seg[np.all(label == rock, axis=-1)] = 0
    label_seg[np.all(label == s, axis=-1)] = 1
    label_seg[np.all(label == sh, axis=-1)] = 2
    label_seg[np.all(label == p, axis=-1)] = 3
    label_seg[np.all(label == ru2, axis=-1)] = 4
    label_seg[np.all(label == ru, axis=-1)] = 5
    label_seg[np.all(label == m, axis=-1)] = 6
    label_seg[np.all(label == k, axis=-1)] = 7
    label_seg[np.all(label == li, axis=-1)] = 8
    label_seg[np.all(label == dh, axis=-1)] = 9
    label_seg[np.all(label == pu, axis=-1)] = 10
    label_seg[np.all(label == th, axis=-1)] = 11
    label_seg[np.all(label == u, axis=-1)] = 12
    label_seg[np.all(label == h, axis=-1)] = 13
    label_seg[np.all(label == le, axis=-1)] = 14
    label_seg[np.all(label == Nhe, axis=-1)] = 15
    # label_seg[np.all(label == ch, axis=-1)] = 16
    # label_seg[np.all(label == thu, axis=-1)] = 17
    label_seg[np.all(label == b, axis=-1)] = 16
    label_seg[np.all(label == ri, axis=-1)] = 17
    label_seg[np.all(label == y, axis=-1)] = 18
    label_seg[np.all(label == shu, axis=-1)] = 19
    label_seg[np.all(label == r, axis=-1)] = 20
    label_seg[np.all(label == ki, axis=-1)] = 21
    label_seg[np.all(label == kadhi, axis=-1)] = 22
    # label_seg[np.all(label == vi, axis=-1)] = 25
    label_seg[np.all(label == g, axis=-1)] = 23
    # label_seg[np.all(label == thi, axis=-1)] = 27
    label_seg[np.all(label == n, axis=-1)] = 24
    label_seg[np.all(label == a, axis=-1)] = 25
    # label_seg[np.all(label == dhi, axis=-1)] = 30
    label_seg[np.all(label == m2, axis=-1)] = 26

    return label_seg


labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))


import random
import numpy as np

image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

############################################################################


n_classes = len(np.unique(labels))

print(n_classes)
from keras.utils import to_categorical

labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

Image_data_gen_args = dict(rotation_range=90, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.3,
                           fill_mode='reflect')

Mask_data_gen_args = dict(rotation_range=90, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.3,
                          fill_mode='reflect',
                          preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype))
image_data_generator = ImageDataGenerator(**Image_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=seed)
image_gen = image_data_generator.flow(X_train, seed=seed)
valid_img_gen = image_data_generator.flow(X_test, seed=seed)

mask_data_generator = ImageDataGenerator(**Mask_data_gen_args)
mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_gen = mask_data_generator.flow(y_train, seed=seed)
valid_maks_gen = mask_data_generator.flow(y_test, seed=seed)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
x_train_1=preprocess_input(X_train)

y_train_1 = preprocess_input(y_train)
image_data_generator_1 = ImageDataGenerator(**Image_data_gen_args)
image_data_generator_1.fit(x_train_1, augment=True, seed=seed)
image_gen_1 = image_data_generator_1.flow(X_train, seed=seed)
valid_img_gen_1 = image_data_generator_1.flow(X_test, seed=seed)

mask_data_generator_1 = ImageDataGenerator(**Mask_data_gen_args)
mask_data_generator.fit(y_train_1, augment=True, seed=seed)
mask_gen_1 = mask_data_generator_1.flow(y_train, seed=seed)
valid_maks_gen_1 = mask_data_generator_1.flow(y_test, seed=seed)

def image_mask_gen(image_gen, mask_gen):
    train_gen = zip(image_gen, mask_gen)
    for (img, mask) in train_gen:
        yield (img, mask)


generator = image_mask_gen(image_gen, mask_gen)

generator_1 = image_mask_gen(image_gen_1,mask_gen_1)
valid_data_gen = image_mask_gen(valid_img_gen, valid_maks_gen)

valid_data_gen_1 = image_mask_gen(valid_img_gen_1,valid_maks_gen_1)
x = image_gen.next()
y = mask_gen.next()
image = x[0]
mask = y[0]

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0])
plt.subplot(1, 2, 2)
plt.imshow(mask[:, :, 0])
plt.show()
#######################################
weights = []
for i in range(0, n_classes):
    weights.append(0.1666)

dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from semantic_model import multi_unet_model, jacard_coef

metrics = ['accuracy', jacard_coef]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
from keras.optimizers import Adam


model.summary()
epchs = 200

checkpoint_filepath = "C:/Users\MuhammedHaleef\OneDrive\Documents\AI & DS/2nd Year\CM2603 DSGP\Final Project\Labelling Dataset\Model_training\models\checkpoints"

lr =[0.00315, 0.0040, 0.0045]
for each in lr:
    class custom_saver(keras.callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs={}):
            if 0 <= epoch < 350:
                self.model.save("C:/Users\MuhammedHaleef\OneDrive\Documents\AI & DS/2nd Year\CM2603 DSGP\Final Project\Labelling Dataset\Model_training\modelsmodel_epoch{}.keras".format(epoch))
    saver = custom_saver(filepath=checkpoint_filepath)
    model.compile(optimizer=Adam(learning_rate=each), loss=total_loss, metrics=metrics)



# class custom_saver_1(keras.callbacks.ModelCheckpoint):
#     def on_epoch_end(self, epoch, logs={}):
#         if epoch>=70:
#             self.model.save("C:/archi/models/without_rock/model_resnet_epoch{}.keras".format(epoch))


# from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0040), loss=total_loss, metrics=metrics)
history1 = model.fit(X_train, y_train,
                     batch_size=16,
                     verbose=1,
                     epochs=200,
                     validation_data=(X_test, y_test),
                     shuffle=False,
                     callbacks=[saver])



checkPoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=False, monitor='val_loss', verbose=1,
                                             save_weights_only=False, mode='max', save_freq=1)
#
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='max',
#     save_best_only=True,save_freq=1
#
# )
saver = custom_saver(filepath=checkpoint_filepath)
# history1 = model.fit(generator,
#                      batch_size=16,
#                      verbose=1,
#                      epochs=100,
#                      steps_per_epoch=100,
#                      validation_steps=50,
#                      validation_data=valid_data_gen,
#                      shuffle=False,callbacks=[saver])

# model.train_on_batch()
# model.save(get_path(i))

# Minmaxscaler
# With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
# With focal loss only, after 100 epochs val jacard is: 0.62  (Mean IoU: 0.6)
# With dice loss only, after 100 epochs val jacard is: 0.74 (Reached 0.7 in 40 epochs)
# With dice + 5 focal, after 100 epochs val jacard is: 0.711 (Mean IoU: 0.611)
##With dice + 1 focal, after 100 epochs val jacard is: 0.75 (Mean IoU: 0.62)
# Using categorical crossentropy as loss: 0.71

##With calculated weights in Dice loss.
# With dice loss only, after 100 epochs val jacard is: 0.672 (0.52 iou)


##Standardscaler
# Using categorical crossentropy as loss: 0.677
import datetime

############################################################
# TRY ANOTHE MODEL - WITH PRETRINED WEIGHTS
# Resnet backbone
# BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)
#
# # preprocess input
# X_train_prepr = preprocess_input(X_train)
# X_test_prepr = preprocess_input(X_test)
#
# # define model
# model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
#
# # compile keras model with defined optimozer, loss and metrics
# # model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
# model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
#
# saver_1 = custom_saver_1(filepath=checkpoint_filepath)
# history2 = model_resnet_backbone.fit(generator_1,steps_per_epoch=100,validation_steps=100,
#
#                                      batch_size=16,
#                                      epochs=100,
#                                      verbose=1,
#                                      validation_data=valid_data_gen_1,shuffle=False,callbacks=[saver_1])

# print(model_resnet_backbone.summary())

# history2 = model_resnet_backbone.fit(X_train_prepr,
#                                      y_train,
#                                      batch_size=16,
#                                      epochs=100,
#                                      verbose=1,
#                                      validation_data=(X_test_prepr, y_test))
# model_resnet_backbone.save("C:/archi/models/with_resnet_backbone.hdf5")

# Minmaxscaler
# With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
# With focal loss only, after 100 epochs val jacard is:
# With dice + 5 focal, after 100 epochs val jacard is: 0.73 (reached 0.71 in 40 epochs. So faster training but not better result. )
##With dice + 1 focal, after 100 epochs val jacard is:
##Using categorical crossentropy as loss: 0.755 (100 epochs)
# With calc. weights supplied to model.fit:

# Standard scaler
# Using categorical crossentropy as loss: 0.74


###########################################################
# plot the training and validation accuracy and loss at each epoch
# history = history1
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# acc = history.history['jacard_coef']
# val_acc = history.history['val_jacard_coef']
#
# plt.plot(epochs, acc, 'y', label='Training IoU')
# plt.plot(epochs, val_acc, 'r', label='Validation IoU')
# plt.title('Training and validation IoU')
# plt.xlabel('Epochs')
# plt.ylabel('IoU')
# plt.legend()
# plt.show()
#
# ##################################
# from keras.models import load_model
#
# model = load_model("C:/archi/models/u-net-standard.hdf5",
#                    custom_objects={'dice_loss_plus_1focal_loss': total_loss,
#                                    'jacard_coef': jacard_coef})
#
# # IOU
# y_pred = model.predict(X_test)
# y_pred_argmax = np.argmax(y_pred, axis=3)
# y_test_argmax = np.argmax(y_test, axis=3)
#
# # Using built in keras function for IoU
# from keras.metrics import MeanIoU
#
# n_classes = 14
# IOU_keras = MeanIoU(num_classes=n_classes)
# IOU_keras.update_state(y_test_argmax, y_pred_argmax)
# print("Mean IoU =", IOU_keras.result().numpy())
#
# #######################################################################
# # Predict on a few images
#
# import random
#
# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth = y_test_argmax[test_img_number]
# # test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input = np.expand_dims(test_img, 0)
# prediction = (model.predict(test_img_input))
# predicted_img = np.argmax(prediction, axis=3)[0, :, :]
#
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img)
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth)
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img)
# plt.show()

#####################################################################
