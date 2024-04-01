import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# from smooth_tiled_predictions import predict_img_with_smooth_windowing
from semantic_model import jacard_coef



def label_to_rgb(predicted_image):

    s = '#ff000f'.lstrip('#')
    s = np.array(tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)))

    sh = '#650006'.lstrip('#')
    sh = np.array(tuple(int(sh[i:i + 2], 16) for i in (0, 2, 4)))

    p = '#0f00ff'.lstrip('#')
    p = np.array(tuple(int(p[i:i + 2], 16) for i in (0, 2, 4)))

    ru2 = '#6713ec'.lstrip('#')
    ru2 = np.array(tuple(int(ru2[i:i + 2], 16) for i in (0, 2, 4)))

    ru = '#070348'.lstrip('#')
    ru = np.array(tuple(int(ru[i:i + 2], 16) for i in (0, 2, 4)))

    m = '#5e5c80'.lstrip('#')
    m = np.array(tuple(int(m[i:i + 2], 16) for i in (0, 2, 4)))

    k = '#8c4747'.lstrip('#')
    k = np.array(tuple(int(k[i:i + 2], 16) for i in (0, 2, 4)))

    li = '#ff03a1'.lstrip('#')
    li = np.array(tuple(int(li[i:i + 2], 16) for i in (0, 2, 4)))

    dh = '#0ffe00'.lstrip('#')
    dh = np.array(tuple(int(dh[i:i + 2], 16) for i in (0, 2, 4)))

    pu = '#7f7ce7'.lstrip('#')
    pu = np.array(tuple(int(pu[i:i + 2], 16) for i in (0, 2, 4)))

    th = '#ff9595'.lstrip('#')
    th = np.array(tuple(int(th[i:i + 2], 16) for i in (0, 2, 4)))

    u = '#681d67'.lstrip('#')
    u = np.array(tuple(int(u[i:i + 2], 16) for i in (0, 2, 4)))

    h = '#f7e500'.lstrip('#')
    h = np.array(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    le = '#00f4ff'.lstrip('#')
    le = np.array(tuple(int(le[i:i + 2], 16) for i in (0, 2, 4)))

    Nhe = '#6e6c93'.lstrip('#')
    Nhe = np.array(tuple(int(Nhe[i:i + 2], 16) for i in (0, 2, 4)))

    ch = '#3b8b9c'.lstrip('#')
    ch = np.array(tuple(int(ch[i:i + 2], 16) for i in (0, 2, 4)))

    thu = '#14593b'.lstrip('#')
    thu = np.array(tuple(int(thu[i:i + 2], 16) for i in (0, 2, 4)))

    b = '#fe8100'.lstrip('#')
    b = np.array(tuple(int(b[i:i + 2], 16) for i in (0, 2, 4)))

    ri = '#4fa681'.lstrip('#')
    ri = np.array(tuple(int(ri[i:i + 2], 16) for i in (0, 2, 4)))

    y = '#b7e29f'.lstrip('#')
    y = np.array(tuple(int(y[i:i + 2], 16) for i in (0, 2, 4)))

    shu = '#dd4d4d'.lstrip('#')
    shu = np.array(tuple(int(shu[i:i + 2], 16) for i in (0, 2, 4)))

    r = '#aa6eff'.lstrip('#')
    r = np.array(tuple(int(r[i:i + 2], 16) for i in (0, 2, 4)))

    ki = '#b81dba'.lstrip('#')
    ki = np.array(tuple(int(ki[i:i + 2], 16) for i in (0, 2, 4)))

    kadhi = '#835812'.lstrip('#')
    kadhi = np.array(tuple(int(kadhi[i:i + 2], 16) for i in (0, 2, 4)))

    vi = '#005c86'.lstrip('#')
    vi = np.array(tuple(int(vi[i:i + 2], 16) for i in (0, 2, 4)))

    g = '#517885'.lstrip('#')
    g = np.array(tuple(int(g[i:i + 2], 16) for i in (0, 2, 4)))

    thi = '#898989'.lstrip('#')
    thi = np.array(tuple(int(thi[i:i + 2], 16) for i in (0, 2, 4)))

    n = '#6d0000'.lstrip('#')
    n = np.array(tuple(int(n[i:i + 2], 16) for i in (0, 2, 4)))

    a = '#718693'.lstrip('#')
    a = np.array(tuple(int(a[i:i + 2], 16) for i in (0, 2, 4)))

    dhi = '#05ff92'.lstrip('#')
    dhi = np.array(tuple(int(dhi[i:i + 2], 16) for i in (0, 2, 4)))

    rock = '#3b4139'.lstrip('#')
    rock = np.array(tuple(int(rock[i:i + 2], 16) for i in (0, 2, 4)))

    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))

    segmented_img[(predicted_image == 0)] = rock
    segmented_img[(predicted_image == 1)] = s
    segmented_img[(predicted_image == 2)] = sh
    segmented_img[(predicted_image == 3)] = p
    segmented_img[(predicted_image == 4)] = ru2
    segmented_img[(predicted_image == 5)] = ru
    segmented_img[(predicted_image == 6)] = m
    segmented_img[(predicted_image == 7)] = k
    segmented_img[(predicted_image == 8)] = li
    segmented_img[(predicted_image == 9)] = dh
    segmented_img[(predicted_image == 10)] = pu
    segmented_img[(predicted_image == 11)] = th
    segmented_img[(predicted_image == 12)] = u
    segmented_img[(predicted_image == 13)] = h
    segmented_img[(predicted_image == 14)] = le
    segmented_img[(predicted_image == 15)] = Nhe
    segmented_img[(predicted_image == 16)] = ch
    segmented_img[(predicted_image == 17)] = thu
    segmented_img[(predicted_image == 18)] = b
    segmented_img[(predicted_image == 19)] = ri
    segmented_img[(predicted_image == 20)] = y
    segmented_img[(predicted_image == 21)] = shu
    segmented_img[(predicted_image == 22)] = r
    segmented_img[(predicted_image == 23)] = ki
    segmented_img[(predicted_image == 24)] = kadhi
    segmented_img[(predicted_image == 25)] = vi
    segmented_img[(predicted_image == 26)] = g
    segmented_img[(predicted_image == 27)] = thi
    segmented_img[(predicted_image == 28)] = n
    segmented_img[(predicted_image == 29)] = a
    segmented_img[(predicted_image == 30)] = dhi

    segmented_img = segmented_img.astype(np.uint8)
    return (segmented_img)
img = cv2.imread("images/prediction1.png", 1)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
prepeocessed= preprocess_input(img)
from keras.models import load_model

model = load_model("model_epoch88.keras", compile=False)
patch_size = 128

# Number of classes
# n_classes = 31
SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
large_img = Image.fromarray(prepeocessed)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
large_img = np.array(large_img)

patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
patches_img = patches_img[:,:,0,:,:,:]
patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, :, :, :]

        # Use minmaxscaler instead of just dividing by 255.
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
            single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :, :]

        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)

patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1],
                                                     patches_img.shape[2], patches_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))#this is the final prediction with labels as the pixels

unique = np.unique(unpatched_prediction)#final prediction
unique_num=[]
for each in unique:
    width = len(unpatched_prediction[0])
    height =len(unpatched_prediction)
    count=0
    for i in range(0,height):
        for j in range(0,width):
            if unpatched_prediction[i][j]==each:
                count+=1
    unique_num.append(count)

for i in range(0,len(unique)):
    print(str(unique[i])+" : "+str(unique_num[i]))
# print(unique)
rgb = label_to_rgb(unpatched_prediction)
plt.imshow(rgb)
plt.axis('off')
plt.show()