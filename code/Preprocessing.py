import cv2
from PIL import Image, ImageEnhance, ImageFilter


def preprocess_image():
    input_image = cv2.imread("C:\\Users\MuhammedHaleef\OneDrive\Documents\AI & DS\\2nd Year\CM2603 DSGP\Final Project\Heritage Decode\images\\raw\image.jpg")

    # Grayscale conversion
    # cv2.imshow('Original', input_image)
    # cv2.waitKey(0)

    # Use the cvtColor() function to grayscale the image
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Grayscale', gray_image)
    # cv2.waitKey(0)

    # # Noise reduction
    # img = gray_image.filter(ImageFilter.MedianFilter)
    #
    # # Contrast enhancement
    # enhancer = ImageEnhance.Contrast(img)
    # enhanced_img = enhancer.enhance(2.0)

    # Save the preprocessed image
    cv2.imwrite("C:\\Users\MuhammedHaleef\OneDrive\Documents\AI & DS\\2nd Year\CM2603 DSGP\Final Project\Heritage Decode\images\preprocessed\images.jpg", gray_image)


preprocess_image()

# from PIL import Image
# import os
#
#
# def read_image(image_name):
#     try:
#         # Construct the full path to the image file
#         image_path = os.path.join("imagefolder", image_name)
#
#         # Open the image file using Pillow
#         img = Image.open(image_path)
#
#         # Display the image (you can replace this with your own processing logic)
#         img.show()
#
#     except Exception as e:
#         print(f"Error: {e}")
#
#
# if __name__ == "__main__":
#     # Replace 'your_image.jpg' with the actual image file name
#     image_name = "your_image.jpg"
#
#     # Check if the image file exists
#     if os.path.exists(os.path.join("imagefolder", image_name)):
#         read_image(image_name)
#     else:
#         print("Image file not found.")
