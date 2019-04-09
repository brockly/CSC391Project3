'''
Alex Brockman
Project #3 Implementation
Apply SIFT function to desired image
'''

import cv2
import numpy as np

#read in image
img = cv2.imread('/Users/Alex/Dropbox/Documents_WakeForest/Junior/CSC 391/Projects/Project #3/Data/Rivers/river_sample_noriver7.jpg')

#convert to gray and apply sift function
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create(nfeatures=500) #nfeatures=n to specify

#create keypoints and draw them on the picture
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(img,kp, None, flags=cv2.DrawMatchesFlags_DEFAULT)

#write the finished result to file
cv2.imwrite('sift_noriver7.jpg',img)