"""
=========================================
Creating Histograms using HOG descriptors
=========================================
Project #3 Implementation
"""

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from PIL import Image
import numpy as np
import cv2

image = '/Users/Alex/Dropbox/Documents_WakeForest/Junior/CSC 391/Projects/Project #3/Data/Threshold/thresh_noriver8.jpg'
image = np.array(Image.open(image), dtype=np.uint8)
print(image.shape)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)

print(fd.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.figure(2)
plt.plot(fd)

plt.show()