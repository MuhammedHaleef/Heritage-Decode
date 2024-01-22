import cv2
from PIL import Image
from skimage import io, color
import matplotlib.pyplot as plt

# original image path
original_image_path = 'C:\\Users\MuhammedHaleef\OneDrive\Documents\AI & DS\\2nd Year\CM2603 DSGP\Final Project\Heritage Decode\images\\raw\image.jpg'
plt.figure(figsize=(50, 50))


# Using OpenCV
original_image_cv2 = cv2.imread(original_image_path)
# Converting to grayscale
gray_image_cv2 = cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2GRAY)

# plotting the original and converted image
plt.subplot(1, 2, 1)
plt.imshow(original_image_cv2[:, :, ::-1])  # OpenCV loads images in BGR format
plt.title('Original (OpenCV)')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gray_image_cv2, cmap='gray')
plt.title('Grayscale (OpenCV)')
plt.axis('off')
plt.show()


# # Using Pillow(PIL)
# original_image_pil = Image.open(original_image_path)
# # Convert to grayscale
# gray_image_pil = original_image_pil.convert('L')
#
# # plotting the original and converted image
# plt.subplot(1, 2, 1)
# plt.imshow(original_image_pil, cmap='gray')
# plt.title('Original (PIL)')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(gray_image_pil, cmap='gray')
# plt.title('Grayscale (PIL)')
# plt.axis('off')
# plt.show()


# # Using scikit-image
# original_image_skimage = io.imread(original_image_path)
# # Convert to grayscale using scikit-image
# gray_image_skimage = color.rgb2gray(original_image_skimage)
#
# # plotting the original and converted image
# plt.subplot(1, 2, 1)
# plt.imshow(original_image_skimage, cmap='gray')
# plt.title('Original (scikit-image)')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(gray_image_skimage, cmap='gray')
# plt.title('Grayscale (scikit-image)')
# plt.axis('off')
# plt.show()


# def preprocess_image():
#     input_image = cv2.imread("C:\\Users\MuhammedHaleef\OneDrive\Documents\AI & DS\\2nd Year\CM2603 DSGP\Final Project\Heritage Decode\images\\raw\image.jpg")
#
#     # Grayscale conversion
#     # cv2.imshow('Original', input_image)
#     # cv2.waitKey(0)
#
#     # Use the cvtColor() function to grayscale the image
#     gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#
#     # cv2.imshow('Grayscale', gray_image)
#     # cv2.waitKey(0)
#
#     # # Noise reduction
#     # img = gray_image.filter(ImageFilter.MedianFilter)
#     #
#     # # Contrast enhancement
#     # enhancer = ImageEnhance.Contrast(img)
#     # enhanced_img = enhancer.enhance(2.0)
#
#     # Save the preprocessed image
#     cv2.imwrite("C:\\Users\MuhammedHaleef\OneDrive\Documents\AI & DS\\2nd Year\CM2603 DSGP\Final Project\Heritage Decode\images\preprocessed\images.jpg", gray_image)
#
#
# preprocess_image()