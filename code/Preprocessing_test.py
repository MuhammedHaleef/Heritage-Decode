import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/28816046/
#displaying-different-images-with-actual-size-in-matplotlib-subplot

def display_image(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width  = im_data.shape[:2]
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    ax.imshow(im_data, cmap='gray')
    plt.show()

# specifying the input(original) image and converted image output paths
image_directory = os.path.join('..', 'images', 'raw')
output_directory = os.path.join('..', 'images', 'preprocessed')

# Get a list of all image files in the specified directory
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # Construct the full path for each image
    image_path = os.path.join(image_directory, image_file)

    try:
        # Read the image using cv2 or perform other operations
        original_image = cv2.imread(image_path)
        width, height = 800, 600
        # resized_image = cv2.resize(original_image, (width, height))
        resized_image = cv2.resize(original_image, (width, height), interpolation=cv2.INTER_AREA)
        edited_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Original Image', resized_image)
        cv2.imshow('Edited Image', edited_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # median_image = cv2.medianBlur(original_image, 3)
        #plt.imshow(cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB))

        # denoising
        # denoised_image = cv2.fastNlMeansDenoisingColored(median_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # saving edited image
        # output_path = os.path.join(output_directory, f'processed_{image_file}')
        # cv2.imwrite(output_path, edited_image)

        # histogram equalization
            # def grayscale(image):
            #     return cv.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # equalized_image = cv2.equalizeHist(gray_image)
            # plt.title('Histogram Equalized Image')
            # plt.imshow(equalized_image, cmap='gray')

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # aequalized_image = clahe.apply(gray_image)
            # plt.imshow(aequalized_image, cmap='gray')
            # plt.title('Adaptive Histogram Equalized Image')


        # gamma correction for contrast adjustment
        # gamma = 1.2
        # adjusted_image = np.clip((aequalized_image / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        # plt.imshow(adjusted_image, cmap='gray')
        # plt.title('Light Adjusted Image')


        # # Define coordinates of the region of interest (ROI)
        # x, y, w, h = 0, 100, 800, 425
        # cropped_image = aequalized_image[y:y + h, x:x + w]
        # plt.imshow(cropped_image, cmap='gray')
        # plt.title('Cropped Image')


        # Apply adaptive thresholding
        # _, thresholded_image = cv2.threshold(cropped_image, 5, 20, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # plt.imshow(thresholded_image, cmap='gray')
        # plt.title('Threshholded Image')

        # thresh, im_bw = cv.threshold(img, 180, 255, cv.THRESH_BINARY)  # TODO: change these values and find the best suited values
        # """thresh - threshold, im_bw - image black & white.
        # first parameter is the image then two integer values are defined as paremeters.
        # cv.THRESH_BINARY - a way of adjusting the threshold, there are multiple ways to ajust the threshold value."""
        # output_path = os.path.join(output_directory, f'Threshold_{image_file}.jpg')
        # cv.imwrite(output_path, im_bw)

        # thresh, im_bw = cv.threshold(gray_image, 140, 255, cv.THRESH_OTSU)

    # thresh, im_bw = cv.threshold(gray_image, 155, 200, cv.THRESH_BINARY_INV)

        # Canny edge
        # edges = cv2.Canny(thresholded_image, 50, 150)
        # plt.imshow(edges, cmap='gray')
        # plt.title('Edge Image')


        # Find contours in the edges
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on a copy of the original image
        # segmented_image = resized_image[y:y + h, x:x + w].copy()
        # cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)

        # plt.imshow(segmented_image)
        # plt.title('Edge_2 Image')


    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")


    # blurred_image_gau = cv2.GaussianBlur(gray_image, (15,11), 0)
    # blurred_image_med = cv2.medianBlur(gray_image, 11)
    # kernel_size = (5, 5)
    # blurred_image_avg = cv2.blur(gray_image, kernel_size)
    #
    # # Define the parameters for the bilateral filter
    # diameter = 9  # Diameter of each pixel neighborhood used during filtering
    # sigma_color = 75  # Filter sigma in the color space
    # sigma_space = 75  # Filter sigma in the coordinate space
    #
    # Apply bilateral filter
    # bilateral_filtered = cv2.bilateralFilter(gray_image, diameter, sigma_color, sigma_space)


    # threshhold_image = applyThreshold(gray_image)
    # cleaned_image = removeSpots(2, threshhold_image[1])

    # plt.imshow(cleaned_image, cmap='gray')
    # plt.title('Noise Reduced Image (OpenCV)')

    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)


# def noise_removal(image):
#     import numpy as np  # working efficiently with numerical data in the memory, cuz the way it uses to store data.
#
#     # Dilation
#     kernel = np.ones((5, 5), np.uint8)  # is an object, 1st argument : size of the kernel/ shape of capturing
#     image = cv.dilate(image, kernel, iterations=1)  # dilating the image, size of the dilation - kernel, 1 pass over the image.
#     # Erosion
#     kernel = np.ones((7, 7), np.uint8)  # can define another kernel for erosion, generally dilation and erosion are done using differnt kernels.
#     image = cv.erode(image, kernel, iterations=1)
#     # Morphing
#     image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)  # morphology and mediun blur are the code lines that get rid of blur mainly in this code segment.
#     # Blur the image
#     image = cv.medianBlur(image, 3)
#     return image
#
#
# for image_file in image_files:
#     image_path = os.path.join(output_directory, f'Threshold_{image_file}.jpg')
#     img = cv.imread(image_path)
#
#     no_noise = noise_removal(img)
#     output_path = os.path.join(output_directory, f'Noise_removed_{image_file}.jpg')
#     cv.imwrite(output_path, no_noise)
#     display_image(output_path)


# def enhance_contrast(image):
#     # Convert to grayscale (usually appropriate for inscriptions)
#     grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     # Apply CLAHE for selective contrast enhancement
#     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
#     enhanced_image = clahe.apply(grayscale_image)
#
#     return enhanced_image
#
# def detect_edges(image):
#     # Perform morphological gradient for focused edge detection
#     # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#     eroded_image = cv.erode(image, kernel)
#     # gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
#
#     # Optionally adjust thresholds for finer control
#     threshold1, threshold2 = 200, 240  # Experiment with suitable values
#     edges = cv.Canny(image, threshold1, threshold2)
#
#     return edges
#
# for image_file in image_files:
#
#     output_path = os.path.join(output_directory, f'gray_scaled_{image_file}.jpg')
#     img = cv.imread(output_path)
#
#     # Preprocess the image
#     enhanced_image = enhance_contrast(img)
#     edges = detect_edges(enhanced_image)
#
#     output_path = os.path.join(output_directory, f'Enhanced_{image_file}.jpg')
#     cv.imwrite(output_path, enhanced_image)
#     display_image(output_path)
#
#     output_path = os.path.join(output_directory, f'Canny_edge_{image_file}.jpg')
#     cv.imwrite(output_path, edges)
#     display_image(output_path)


# except FileNotFoundError as e:
#     print(f"Error: {e}")
# except cv2.error as e:
#     print(f"OpenCV Error: {e}")
# except OSError as e:
#     print(f"OS Error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")




# comparing two images
# from skimage.metrics import structural_similarity as ssim

# # Calculate Structural Similarity Index (SSI)
# ssi_index, _ = ssim(gray_image1, gray_image2, full=True)
# print(f"SSI Index: {ssi_index}")

# Calculate Mean Squared Error (MSE)
# mse = np.sum((gray_image1 - gray_image2) ** 2) / float(gray_image1.shape[0] * gray_image1.shape[1])
# print(f"Mean Squared Error (MSE): {mse}")

