# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint
#
# # Load MobileNetV2 base model (pre-trained on ImageNet)
# base_model = MobileNetV2(input_shape=(None, None, 3), include_top=False, weights='imagenet')
#
# # Add a segmentation head on top of the base model
# x = base_model.output
# x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)  # Output layer for binary segmentation
#
# # Create the segmentation model
# segmentation_model = Model(inputs=base_model.input, outputs=x)
#
# # Compile the model (adjust the learning rate and other parameters as needed)
# segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Optional: Freeze layers in the base model to retain pre-trained weights
# for layer in base_model.layers:
#     layer.trainable = False
#
#
# # Assuming you have loaded and preprocessed your dataset
# train_data_dir = '././images/raw/raw_image.jpg'
# val_data_dir = 'path/to/validation_data'
# batch_size = 32
# image_size = (256, 256)
#
# # Create data generators with augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# val_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='input',  # Use 'input' since it's an image segmentation task
#     color_mode='rgb',  # or 'grayscale' based on your dataset
#     shuffle=True
# )
#
# validation_generator = val_datagen.flow_from_directory(
#     val_data_dir,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='input',
#     color_mode='rgb',
#     shuffle=False
# )
#
# # Fine-tune the model
# num_epochs = 10  # Adjust based on your requirements
# checkpoint_filepath = 'code/segmentation_model.h5'
#
# # Set up a model checkpoint to save the best model during training
# model_checkpoint = ModelCheckpoint(
#     checkpoint_filepath,
#     save_best_only=True,
#     monitor='val_loss',
#     mode='min',
#     verbose=1
# )
#
# # Train the model
# history = segmentation_model.fit(
#     train_generator,
#     epochs=num_epochs,
#     validation_data=validation_generator,
#     callbacks=[model_checkpoint]
# )
#
# # Save the trained model
# segmentation_model.save('code/segmentation_model.h5')


# from tensorflow.keras.applications import MobileNetV2
#
# # Load pre-trained MobileNetV2 with ImageNet weights
# pretrained_mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False)
#
# # Save the pre-trained model to an HDF5 file
# pretrained_mobilenetv2.save('segmentation_model.h5')

# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
# from tensorflow.keras.models import Model
#
# # Load pre-trained MobileNetV2 model
# base_model = MobileNetV2(weights='imagenet', include_top=False)
#
# # Add segmentation head on top of MobileNetV2
# x = base_model.output
# x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D(size=(2, 2))(x)
# x = Concatenate()([x, base_model.get_layer('block_6_expand_relu').output])  # Adjust layer name based on MobileNetV2 architecture
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D(size=(2, 2))(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D(size=(2, 2))(x)
# segmentation_output = Conv2D(1, (1, 1), activation='sigmoid')(x)  # Output layer for binary segmentation
#
# # Create the segmentation model
# segmentation_model = Model(inputs=base_model.input, outputs=segmentation_output)
#
# # Save the segmentation model to an HDF5 file
# segmentation_model.save('path/to/save/segmentation_model.h5')
import tensorflow as tf
import deeplabv3plus as dlv3

# Load the DenseNet121 model from TensorFlow Hub
model = tf.keras.applications.densenet.DenseNet121(weights="imagenet", include_top=False)

# Convert the pre-trained model to DeepLabV3+
model = dlv3.DeepLabV3Plus(model, classes=21)  # Adjust the number of classes as needed

# Use the model for inference
# predictions = model.predict(your_image_tensor)


# Use DeepLabV3+ ResNet for accuracy
# model = tf.keras.applications.DeepLabV3Plus(weights='imagenet')

# If you have a pre-trained ResNet model, adjust this path
model_path = "path/to/your/resnet_model.h5"
if model_path:
    model.load_weights(model_path)

# Pre-processing function (adjust based on your input format)
def preprocess_image(image):
    # Resize to expected input size (may differ for other variants)
    image = tf.image.resize(image, (512, 512))
    # Normalize as needed for ResNet (typically between 0-1)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Define conversion parameters (adjust based on tool and requirements)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Prioritize accuracy for now
converter.target_spec.supported_types = [tf.float32]  # Maintain float32 for full precision

# Quantization might compromise accuracy, consider skipping for now
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.quantization_parameters = {
#     '': tf.lite.QuantizationParameters(
#         scales=[1/255.0],
#         zero_points=[0]
#     )
# }

# Convert the model (may take longer due to larger model size)
tflite_model = converter.convert()

# Save the converted model (.tflite)
with open("deeplabv3_resnet_mobile.tflite", "wb") as f:
    f.write(tflite_model)

print("DeepLabV3+ ResNet model converted and saved for mobile integration!")


predictions = model.predict(your_image_tensor)
