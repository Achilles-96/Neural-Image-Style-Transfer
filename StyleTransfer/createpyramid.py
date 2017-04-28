import cv2
import numpy as np
import sys

base_path1 = 'InputContentImages/'
image_sel = sys.argv[1]

A = cv2.imread(base_path1 + image_sel + '.jpg')
G = A.copy()

# Create the pyramids
gauss_A = [G]
lap_A = []

levels = 4

for i in xrange(levels-1):
    G = cv2.pyrDown(G)
    gauss_A.append(G)

lap_A.append(gauss_A[len(gauss_A)-1])

for i in xrange(levels-1,0,-1):
    gauss_upscl = cv2.pyrUp(gauss_A[i])
    gauss_upscl = cv2.resize(gauss_upscl, (gauss_A[i-1].shape[1], gauss_A[i-1].shape[0]))
    lap = cv2.subtract(gauss_A[i-1], gauss_upscl)
    lap = lap * 2
    lap = cv2.dilate(lap, np.ones((3,3), np.uint8))
    lap_A.append(lap)

base_path2 = 'InputContentImagePyramids/'

for i in xrange(levels):
    cv2.imwrite(base_path2 + image_sel[:-4] + 'g' + str(i) + '.jpg', cv2.resize(gauss_A[i], (A.shape[1], A.shape[0])))
    cv2.imshow('Levels Gauss' + str(i), cv2.resize(gauss_A[i], (A.shape[1], A.shape[0])))
    cv2.waitKey(0)

for i in xrange(levels):
    cv2.imwrite(base_path2 + image_sel[:-4] + 'l' + str(i) + '.jpg', cv2.resize(lap_A[i], (A.shape[1], A.shape[0])))
    cv2.imshow('Levels Lap' + str(i), cv2.resize(lap_A[i], (A.shape[1], A.shape[0])))
    cv2.waitKey(0)

cv2.destroyAllWindows()