import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img1 = cv2.imread('./dataset/IMG_8102.png')
img1_gray = cv2.imread('./dataset/IMG_8102.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

detector = cv2.SimpleBlobDetector()
keypoints = detector.detect(img1)

print(keypoints)
