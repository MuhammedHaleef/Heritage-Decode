import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np

# Load the two images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Convert images to grayscale (optional, but can be useful for SSIM)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Calculate Structural Similarity Index (SSI)
ssi_index, _ = ssim(gray_image1, gray_image2, full=True)
print(f"SSI Index: {ssi_index}")

# Calculate Mean Squared Error (MSE)
mse = np.sum((gray_image1 - gray_image2) ** 2) / float(gray_image1.shape[0] * gray_image1.shape[1])
print(f"Mean Squared Error (MSE): {mse}")

# Visualize the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
ax1.set_title('Image 1')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
ax2.set_title('Image 2')
ax2.axis('off')

plt.show()