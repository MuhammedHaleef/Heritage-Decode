from tensorflow.keras.applications import unet, deeplabv3plus

# Load U-Net pre-trained model
unet_model = unet.Unet(input_shape=(256, 256, 3), classes=1, activation='sigmoid')
unet_model.summary()

# Load DeepLabV3 pre-trained model
deeplab_model = deeplabv3.DeepLabV3(input_shape=(256, 256, 3), classes=1, activation='sigmoid')
deeplab_model.summary()

# # Load Mask R-CNN pre-trained model
# maskrcnn_model = mask_rcnn.MaskRCNN(mode='training', config=None, model_dir='./')
# maskrcnn_model.summary()