"""
Created on Tue Jun 30 17:17:41 2020

@author: Yohan Reyes
"""

# =============================================================================
# %% Vars
# =============================================================================

fast_slow = 'f'

# =============================================================================
# %% imports
# =============================================================================
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from sklearn.cluster import KMeans

# =============================================================================
# %% Multithreads
# =============================================================================
 
# speed-up using multithreads
cv2.setUseOptimized(True);
cv2.setNumThreads(4);



# =============================================================================
# %% Load Image
# =============================================================================

# Scaling the image pixels values within 0-1
img = imread('photo.jpg') / 255
im = cv2.imread('photo.jpg')
plt.imshow(img)
plt.title('Original')
plt.show()
newHeight,newWidth,_ = im.shape

# =============================================================================
# %% Cluster Image Segmentations
# =============================================================================

# For clustering the image using k-means, we first need to convert it into a 2-dimensional array
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

# Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image
# tweak the cluster size and see what happens to the Output
kmeans = KMeans(n_clusters=5, random_state=0).fit(image_2D)
clustered = kmeans.cluster_centers_[kmeans.labels_]
# Reshape back the image from 2D to 3D image
clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Clustered Image')
plt.show()

# =============================================================================
# %% Region Proposal Via Selective Search 
# =============================================================================

# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
# set input image on which we will run segmentation
ss.setBaseImage(im)
 
# Switch to fast but low recall Selective Search method
if (fast_slow == 'f'):
    ss.switchToSelectiveSearchFast()
# Switch to high recall but slow Selective Search method
elif (fast_slow == 'q'):
    ss.switchToSelectiveSearchQuality()
# if argument is neither f nor q print help message
else:
    print(__doc__)

 
# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
 
# number of region proposals to show
numShowRects = 100
# increment to increase/decrease total number
# of reason proposals to be shown
increment = 50
 
while True:
    # create a copy of original image
    imOut = im.copy()
 
    # itereate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break
 
    # show output
    cv2.imshow("Output", imOut)
 
    # record key press
    k = cv2.waitKey(0) & 0xFF
 
    # m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    # q is pressed
    elif k == 113:
        break
# close image show window
cv2.destroyAllWindows()




# =============================================================================
# %% END
# =============================================================================
