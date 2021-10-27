import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from skimage import data, feature
import os

def is_flare(pixel):
    if pixel[0]>= 80 and pixel[0]<= 190 and pixel[1] >= 0.1 and pixel[2] >= 150:
        return True
    else:
        return False
    
def flare_list(img1):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blobs_dog = feature.blob_dog(img1_gray, threshold=0.1, max_sigma=20)
    blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)

    a = []
    img1_hsv = cv2.cvtColor(img1.astype('float32'), cv2.COLOR_BGR2HSV)
    for i in blobs_dog:
        max_pixel = []
        max_value = -1
        clip = img1[int(i[0]-i[2]):int(i[0]+i[2]), int(i[1]-i[2]):int(i[1]+i[2])]
        clip_hsv = img1_hsv[int(i[0]-i[2]):int(i[0]+i[2]), int(i[1]-i[2]):int(i[1]+i[2])]
        for x in range(clip.shape[0]):
            for y in range(clip.shape[1]):
                if clip[x][y].sum() >= max_value:
                    max_value = clip[x][y].sum()
                    max_pixel = clip_hsv[x][y]
    
        if max_value != -1:
            if is_flare(max_pixel):
                a.append(i)
    a=np.array(a)
    return a
    
'''
data = os.listdir('./dataset')

for d in data:
    img1 = cv2.imread('./dataset/'+d)
    #img1_gray = cv2.imread('./dataset/'+d, cv2.IMREAD_GRAYSCALE)
    
    blobs_dog = feature.blob_dog(img1_gray, threshold=0.1, max_sigma=20)
    blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)

    a = []
    img1_hsv = cv2.cvtColor(img1.astype('float32'), cv2.COLOR_BGR2HSV)
    for i in blobs_dog:
        max_pixel = []
        max_value = -1
        clip = img1[int(i[0]-i[2]):int(i[0]+i[2]), int(i[1]-i[2]):int(i[1]+i[2])]
        clip_hsv = img1_hsv[int(i[0]-i[2]):int(i[0]+i[2]), int(i[1]-i[2]):int(i[1]+i[2])]
        for x in range(clip.shape[0]):
            for y in range(clip.shape[1]):
                if clip[x][y].sum() >= max_value:
                    max_value = clip[x][y].sum()
                    max_pixel = clip_hsv[x][y]
    
        if max_value != -1:
            if is_flare(max_pixel):
                a.append(i)
    a=np.array(a)

    output_img = img1.copy()
    for i in a:
        output_img = cv2.circle(output_img, (int(i[1]),int(i[0])), round(i[2]), color=(255, 255, 0), thickness = 1)
    cv2.imwrite('./result/flare_'+d,output_img)
'''