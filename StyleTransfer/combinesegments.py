import cv2
import numpy as np
import sys

content_file = sys.argv[1]
style_file = sys.argv[2]

content_img_path = 'InputContentImages/' + content_file + '.jpg'
content_img = cv2.imread(content_img_path)

result_file = content_file + '_' + style_file

kernel = np.ones((3,3),np.float32)/9

result_imgs = []
for i in range(6):
    result_imgs.append(cv2.resize(cv2.imread('Results/' + result_file + '_' + str(i) + '.jpg'), (content_img.shape[1], content_img.shape[0])))

content_map_path = 'InputContentImageMasks/' + content_file + '_mask.jpg'
content_map = cv2.resize(cv2.imread(content_map_path), (content_img.shape[1], content_img.shape[0]))

style_map_path = 'InputStyleImageMasks/' + style_file + '_mask.jpg'
style_map = cv2.resize(cv2.imread(style_map_path), (content_img.shape[1], content_img.shape[0]))

style_output_dir = 'InputStyleSegments/'
content_output_dir = 'InputContentSegments/'

# Do not use black as a element of the mask, black color regions in the mask will be ignored.

cont_shape = content_img.shape

# BGR color scheme
map_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]]

segment_num = 0

final_img = content_img.copy()

for color in map_colors:
    for x in xrange(0, cont_shape[0]):
        for y in xrange(0, cont_shape[1]):
            if np.abs(content_map[x][y][0] - color[0]) <= 10 and np.abs(content_map[x][y][1] - color[1]) <= 10 and np.abs(content_map[x][y][2] - color[2]) <= 10:
                final_img[x][y] = result_imgs[segment_num][x][y]
    segment_num += 1

kernel1 = np.ones((3,3),np.float32)/9
cv2.imwrite(content_file + '_' + style_file + '.jpg', cv2.filter2D(final_img,-1,kernel1))