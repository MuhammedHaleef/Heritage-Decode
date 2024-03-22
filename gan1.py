import numpy as np
import os
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

# Define parameters
latent_dim = 128
height = 64
width = 64
channels = 3
batch_size = 10
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available")
    # Limit GPU memory growth to prevent allocation error
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU is not available")

# Generator model
generator = keras.Sequential([
    keras.Input(shape=(latent_dim,)),
    layers.Dense(8 * 8 * 256),
    layers.Reshape((8, 8, 256)),
    layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(channels, kernel_size=5, strides=2, padding="same", activation="tanh"),
])

# Discriminator model
discriminator = keras.Sequential([
    keras.Input(shape=(height, width, channels)),
    layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])

# Compile the discriminator
discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Combine generator and discriminator into a GAN
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Directory containing images
image_dir = 'C:/archi/train5'

# Image data generator
image_generator = ImageDataGenerator(preprocessing_function=lambda x: (x.astype("float32") - 127.5) / 127.5)
dataset = image_generator.flow_from_directory(
    image_dir,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    subset=None
)
save_dir ="C:/archi/w11"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Training loop
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print(epoch)
    print("k")
    count=0
    for batch in dataset:
        # Train discriminator
        print(count)
        count+=1
        noise = np.random.normal(0, 1, (batch.shape[0], latent_dim))
        fake_images = generator.predict(noise)
        combined_images = np.concatenate([batch, fake_images])
        labels = np.concatenate([np.ones((batch.shape[0], 1)), np.zeros((batch.shape[0], 1))])
        labels += 0.05 * np.random.random(labels.shape)
        discriminator_loss = discriminator.train_on_batch(combined_images, labels)

        # Train generator
        noise = np.random.normal(0, 1, (batch.shape[0], latent_dim))
        misleading_labels = np.zeros((batch.shape[0], 1))
        generator_loss = gan.train_on_batch(noise, misleading_labels)

    # Print losses
    print(f"Discriminator loss: {discriminator_loss}")
    print(f"Generator loss: {generator_loss}")
    if epoch % 10 == 0:
        for i, image in enumerate(fake_images):
            generated_image = tf.keras.preprocessing.image.array_to_img(image * 127.5 + 127.5)
            generated_image.save(os.path.join(save_dir, f"generated_image_epoch_{epoch}_sample_{i}.png"))
