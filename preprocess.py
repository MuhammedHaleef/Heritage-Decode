import cv2
import numpy as np


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