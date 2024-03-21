import numpy as np
from scipy import io
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


#
# img = cv2.imread(image_path,3)
# blur = cv2.GaussianBlur(img, (15, 15), 0)
# cv2.imwrite('Filtered_Image.png', blur)
# display_image('Filtered_Image.png')
#
# b, g, r = cv2.split(img)
# rgb_img = cv2.merge([r,g,b])
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# # noise removal
# kernel = np.ones((2, 2), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
# # sure background area
# sure_bg = cv2.dilate(closing, kernel, iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
# # Threshold
# ret , sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255,0)
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


# #-------------------------histogram using calculation
# # find frequency of pixels in range 0-255
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img= cv2.imread(image_path,3)
# histr = cv2.calcHist([img],[0],None,[256],[0,256])
# # show the plotting graph of an image
# plt.plot(histr)
# plt.title("Histogram of image"), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.show()
# # alternative way to find histogram of an image
# plt.hist(img.ravel(),256,[0,256])
# plt.title("GENERATING HISTOGRAM OF ORIENTED GRADIENTS"), plt.xticks([]), plt.yticks([])
# plt.show()