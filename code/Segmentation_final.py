import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def display_image(im_path):     # to get an idea of the pre-processing technique
    dpi = 80
    im_data = plt.imread(im_path)
    height, width = im_data.shape[:2]
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


def enhance_contrast(image):
    # Convert to grayscale (usually appropriate for inscriptions)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for selective contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    enhanced_image = clahe.apply(grayscale_image)

    return enhanced_image


raw_image_directory = os.path.join('..', 'images', 'raw')
output_directory = os.path.join('..', 'images', 'final_testing')
# preprocessed_image_directory = os.path.join('..', 'images', 'preprocessed')
image_path = os.path.join(raw_image_directory, "Mihinthale.jpg")
# display_image(image_path)

# image_path = '../images/raw/Mihinthale.jpg'


def save_image(edited_image, file_name):
    output_path = os.path.join(output_directory, file_name)
    cv2.imwrite(output_path, edited_image)


def image_preprocessing(img_path):
    img = cv2.imread(img_path)  # original image
    img_copy = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    # print(height, width)

    # Resize image for a standard size
    resize_img = cv2.resize(img_copy, dsize=(1320, int(1320 * height / width)), interpolation=cv2.INTER_AREA)
    save_image(resize_img, 'resized_image.png')

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    save_image(blur, 'Gau_blur.png')
    # median_image = cv2.medianBlur(img_copy, 3)

    b, g, r = cv2.split(img_copy)
    rgb_img = cv2.merge([r, g, b])
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
    save_image(sure_bg, 'dil.png')

    edge = cv2.Canny(sure_bg, 2, 5)
    save_image(edge, 'canny_edges.png')

    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]
    #
    # plt.subplot(211), plt.imshow(rgb_img)
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(212), plt.imshow(thresh, 'gray')
    # plt.imsave(r'thres_hold.png', thresh)
    # plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    #
    # plt.subplot(211), plt.imshow(closing, 'gray')
    # plt.title("morphologyEx:Closing:2x2"), plt.xticks([]), plt.yticks([])
    # plt.subplot(212), plt.imshow(sure_bg, 'gray')
    # plt.imsave(r'dil_a.tion.png', sure_bg)
    # plt.title("Dilation"), plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    #
    # plt.subplot(211), plt.imshow(dist_transform, 'gray')
    # plt.title("Distance Transform"), plt.xticks([]), plt.yticks([])
    # plt.subplot(212), plt.imshow(sure_fg, 'gray')
    # plt.title("Thresholding"), plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    #
    # plt.subplot(211), plt.imshow(unknown, 'gray')
    # plt.title("Unknown"), plt.xticks([]), plt.yticks([])
    # plt.subplot(212), plt.imshow(img, 'gray')
    # plt.title("Result from Watershed"), plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()

    # ________________________________________________________________________________________________________________

    # save_image(blur, 'median_image.png')
    # de_noised_image = cv2.fastNlMeansDenoisingColored(blur, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # save_image(de_noised_image, 'fastdenoised_image.png')
    #
    # # Grayscale
    # gray_img = cv2.cvtColor(de_noised_image, cv2.COLOR_BGR2GRAY)
    # save_image(gray_img, 'gray_image.png')
    #
    # # Histogram Equalize Image
    # equalized_image = cv2.equalizeHist(gray_img)
    # save_image(equalized_image, 'hist_equalize_image.png')
    #
    # # Adaptive Histogram Equalized Image
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # aequalized_image = clahe.apply(gray_img)
    # save_image(aequalized_image, 'adapt_hist_equalize_image.png')
    #
    # # Light adjusted image
    # # Assuming gamma correction for contrast adjustment
    # gamma = 1.2
    # adjusted_image = np.clip((aequalized_image / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    # save_image(adjusted_image, 'light_adjusted_image.png')
    #
    # # Apply adaptive thresholding
    # _, threshold_image = cv2.threshold(aequalized_image, 5, 20, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # _, threshold_image = cv2.threshold(aequalized_image, 50, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # save_image(threshold_image, 'threshold_image.png')
    # # plt.imshow(threshold_image, cmap='gray')
    # # plt.title('threshold_image ')
    #
    # # thresh, im_bw = cv.threshold(img, 180, 255, cv.THRESH_BINARY)  # TODO: change these values and find the best suited values
    # # """thresh - threshold, im_bw - image black & white.
    # # first parameter is the image then two integer values are defined as paremeters.
    # # cv.THRESH_BINARY - a way of adjusting the threshold, there are multiple ways to ajust the threshold value."""
    # # output_path = os.path.join(output_directory, f'Threshold_{image_file}.jpg')
    # # cv.imwrite(output_path, im_bw)
    #
    # # thresh, im_bw = cv.threshold(gray_image, 140, 255, cv.THRESH_OTSU)
    #
    # # thresh, im_bw = cv.threshold(gray_image, 155, 200, cv.THRESH_BINARY_INV)
    #
    # # Canny edge
    # edges = cv2.Canny(threshold_image, 50, 150)
    # # plt.imshow(edges, cmap='gray')
    # # plt.title('Edge Image')
    #
    # # Find contours in the edges
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Draw the contours on a copy of the original image
    # cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
    #
    # # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)
    # # bin_img1 = thresh_img.copy()
    # # # bin_img2 = bin_img.copy()
    # # # output_path = os.path.join(output_directory, "binImg_Mihinthale(2).jpg")
    # # # cv2.imwrite(output_path, bin_img1)  # not saving the filtered image
    # #
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    # #
    # # final_thr = cv2.morphologyEx(bin_img1, cv2.MORPH_CLOSE, kernel)
    # # # output_path = os.path.join(output_directory, "final_Mihinthale(2).jpg")
    # # cv2.imwrite(output_path, final_thr)  # not saving the filtered image
    # # contr_retrival = final_thr.copy()


image_preprocessing(image_path)
