"""
Project #3 Thresholding
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

#Import image and apply the threshold
img = cv2.imread('/Users/Alex/Dropbox/Documents_WakeForest/Junior/CSC 391/Projects/Project #3/Data/Rivers/river_sample_noriver10.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(thresh,(5,5),0)
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#Show the modified image and save it to file
cv2.imshow('image',thresh)
#cv2.imshow('image2',blur)
cv2.imwrite('thresh_noriver10.jpg',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()