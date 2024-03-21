import sys
import cv2
import os
import numpy as np

sys.setrecursionlimit(10 ** 6)


def image_segmentation():
    # image thresholding
    try:
        image_directory = os.path.join('..', 'images', 'raw')
        output_directory = os.path.join('..', 'images', 'segmented')
        image_path = os.path.join(image_directory, "Mihinthale(2).jpg")
        src_img = cv2.imread(image_path)

        copy = src_img.copy()
        height = src_img.shape[0]
        width = src_img.shape[1]
        # print(height, width)

        src_img_copy = src_img.copy()

        src_img = cv2.resize(copy, dsize=(1320, int(1320 * height / width)), interpolation=cv2.INTER_AREA)

        height = src_img.shape[0]
        width = src_img.shape[1]
        # print(height, width)


        bin_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)
        bin_img1 = bin_img.copy()
        bin_img2 = bin_img.copy()
        output_path = os.path.join(output_directory, "binImg_Mihinthale(2).jpg")
        cv2.imwrite(output_path, bin_img1)  # not saving the filtered image

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)

        final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        output_path = os.path.join(output_directory, "final_Mihinthale(2).jpg")
        cv2.imwrite(output_path, final_thr)  # not saving the filtered image
        contr_retrival = final_thr.copy()


        # character segmentation
        chr_img = cv2.imread("Edged_Image.png")

        contours, hierarchy = cv2.findContours(chr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_height = src_img.shape[0]
        new_width = src_img.shape[1]

        resized_image = cv2.resize(src_img_copy, (new_width, new_height))

        edges = []
        # print(len(contours))

        # getting x1, x2, y1, y2 edges of a letter for crop
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                temp = []
                temp.append(x)
                temp.append(y)
                temp.append(w)
                temp.append(h)
                edges.append(temp)
                cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # reordering according to x index of edge for saving cropped image in oder
        n = len(edges)

        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if edges[j][0] > edges[j + 1][0]:
                    edges[j], edges[j + 1] = edges[j + 1], edges[j]



        # cropping original image form letter edges
        i = 0
        for edge in edges:
            crop = resized_image[edge[1]:(edge[1] + edge[3]), edge[0]:(edge[0] + edge[2])]
            crop = cv2.resize(crop, (224, 224))
            output_path = os.path.join(output_directory, "test_Mihinthale(2).jpg")
            cv2.imwrite(output_path, crop)  # not saving the filtered image
            # cv2.imwrite("segmentation_module/segmented_letters/crop_{0}.jpg".format(i), crop)
            i = i + 1

        # /character segmentation

        return True

    except:
        return False


image_segmentation()
import os
import cv2
from matplotlib import pyplot as plt

image_directory = os.path.join('..', 'images', 'raw')
output_directory = os.path.join('..', 'images', 'segmented')
image_path = os.path.join(image_directory, "Mihinthale.jpg")
src_img = cv2.imread(image_path)

copy = src_img.copy()
height = src_img.shape[0]
width = src_img.shape[1]
# print(height, width)

src_img_copy = src_img.copy()

src_img = cv2.resize(copy, dsize=(1320, int(1320 * height / width)), interpolation=cv2.INTER_AREA)

chr_img = cv2.imread("Edged_Image.png")
chr_img = cv2.cvtColor(chr_img, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(chr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imwrite('Contour_image.png', cv2.drawContours(src_img, contours, -1, (0, 255, 0), 1))
# # Define coordinates of the region of interest (ROI)
# x, y, w, h = 0, 100, 800, 425
# # Draw the contours on a copy of the original image
# segmented_image = src_img[y:y+h, x:x+w].copy()


# plt.imshow(src_img)
# plt.title('Edge_2 Image')
# plt.axis('off')
# plt.show()

import Segmentation_final as segmentation

segmentation.display_image('Contour_image.png')
