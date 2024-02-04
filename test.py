import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def findedges(image_input):
    pass
    fft = np.fft.fft2(image_input)
    f_shift = np.fft.fftshift(fft)
    # magnitude = 20*np.log(np.abs(f_shift))

    rows = image_input.shape[0]
    coloumns = image_input.shape[1]

    center_row = rows // 2
    center_col = coloumns // 2
    f_shift[center_row - 30: center_row + 30, center_col - 30:center_col + 30] = 0

    f_inv_shift = np.fft.ifftshift(f_shift)

    image_edges = np.fft.ifft2(f_inv_shift)
    image_edges = np.abs(image_edges)

    return image_edges

def getneighbours1(limits, i , j,input,end):
    ranges = []
    neib = []
    for c in range(-limits, limits + 1):
        ranges.append(c)

    try:
        for r in ranges:
            for k in ranges:
                if end>=(i+r)>=0 and end>=(j+k)>=0:
                    neib.append(input[i+r][j+k])
    except Exception as e:
        pass
    return neib


def clean(limit, input):
    for i in range(0, len(input)):
        row = input[i]
        for l in range(0, len(row)):
            nei = getneighbours1(limit, i, l,input,80)
            spots=0
            for r in nei:
                if r==255.0:
                    spots+=1
            if spots<10:
                input[i][l]=0
    return input



def applyThreshold(image_input):
    threshold_value = 77

    image_input = cv2.resize(image_input, (100, 100))

    thresh_image = cv2.threshold(image_input, threshold_value, 255.0, cv2.THRESH_BINARY)

    return thresh_image
def clean_large(image_input,limit,cuttOff):
    for i in range(0, len(image_input)):
        row = image_input[i]
        for l in range(0, len(row)):
            nei = getneighbours1(limit, i, l,image_input,80)
            spots=0
            for r in nei:
                if r==255.0:
                    spots+=1
            if spots>cuttOff:
                image_input[i][l]=0
    return image_input


path ="C:/archi/test/g(1).png"
path2="C:/archi/real/DSC07458.JPG"
image  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
plt.subplot(1,5,1)
plt.imshow(image,cmap="gray")

thresh = applyThreshold(image)[1]
plt.subplot(1,5,2)
plt.imshow(thresh,cmap="gray")

cleaned = clean(2,thresh)
plt.subplot(1,5,3)
plt.imshow(cleaned,cmap="gray")

large_clean = clean_large(cleaned,20,380)
plt.subplot(1,5,4)
plt.imshow(large_clean,cmap="gray")

edges = findedges(cleaned)
plt.subplot(1,5,5)
plt.imshow(edges,cmap="gray")
plt.show()

print(edges)
print("d")
model = load_model("C:/archi/models/model1.keras")
ed=cv2.resize(edges,(80,80))
ed1 = np.expand_dims(ed,axis=-1)
ed2 = np.repeat(ed1,3,axis=-1)
print("w")
predictions = model.predict(np.expand_dims(ed2,axis=0))
print(predictions)

# image = cv2.imread(path2)
# plt.imshow(image)
# plt.show()




