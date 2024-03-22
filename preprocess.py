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

    image_input = cv2.resize(image_input, (80, 80))

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


def in_visited(visited, i, j):
    if_visited = False
    for each in visited:
        if each[0] == i and each[1] == j:
            if_visited = True
    return if_visited
def clean_round(input, limit_high, limit_low):
    for i in range(0, len(input)):
        # print("+ i "+str(i)+"/"+str(len(input)))

        row = input[i]
        for j in range(0, len(row)):
            # if i == 5 and j == 18:
            #     print("j")

            # print("j " + str(j) + "/" + str(len(row)))
            count = 0
            found = False
            visited = []
            start = [i, j]
            current_node = []
            current_node_i=i
            current_node_j=j
            # if current_node[0] == 10 and current_node[1] == 17:
            #     print("g")
            # visited.append([start[0],start[1]])
            while count < limit_high or found == False:
                count1 = 0
                neibours = getneighbours1(1, current_node_i, current_node_j ,len(input), len(row))
                for Node in neibours:
                    count1+=1
                    if input[Node[0]][Node[1]] == True:
                        if count > 2 and in_visited(visited, Node[0], Node[1]) == False:
                            count += 1
                            visited.append([Node[0], Node[1]])
                            current_node.clear()
                            current_node_i=Node[0]
                            current_node_j=Node[1]
                            if Node[0] == start[0] and Node[1] == start[1]:
                                found = True
                                count += limit_high

                            break
                        elif count <= 2 and in_visited(visited, Node[0], Node[1]) == False:
                            if Node[0] != start[0] and Node[1] != start[1]:

                                count += 1
                                visited.append([Node[0], Node[1]])
                                # current=[each[0],each[1]]
                                current_node.clear()
                                current_node_i=Node[0]
                                current_node_j=Node[1]

                                break
                if count1==len(neibours):
                    break
                if found == True:
                    for Node1 in visited:
                        input[Node1[0]][Node1[1]] = False
    return input

def normalize(input):
    for i in range(0,len(input)):
        row= input[i]
        for j in range(0,len(row)):
            value = input[i][j]
            input[i][j]=value*255

    return input
