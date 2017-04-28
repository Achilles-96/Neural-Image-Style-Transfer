import cv2
import numpy as np
import sys

sketch_file = sys.argv[1]
result_file = sys.argv[2]

sketch_img = cv2.imread(sketch_file)
result_img = cv2.imread(result_file)

cont_shape = result_img.shape

final_img = result_img.copy()

for x in xrange(0, cont_shape[0]):
    for y in xrange(0, cont_shape[1]):
        if sketch_img[x][y][0] <= 30 and sketch_img[x][y][1] <= 30 and sketch_img[x][y][2] <= 30:
            final_img[x][y] = sketch_img[x][y]

cv2.imwrite(sys.argv[3], final_img)