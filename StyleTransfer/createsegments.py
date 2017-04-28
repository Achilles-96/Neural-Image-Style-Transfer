import cv2
import numpy as np
import sys

content_file = sys.argv[1]
content_img_path = 'InputContentImages/' + content_file + '.jpg'
content_img = cv2.imread(content_img_path)

style_file = sys.argv[2]
style_img_path = 'InputStyleImages/' + style_file + '.jpg'
style_img = cv2.resize(cv2.imread(style_img_path), (content_img.shape[1], content_img.shape[0]))

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

for color in map_colors:
    new_content_img = content_img.copy()
    new_style_img = style_img.copy()
    for x in xrange(0, cont_shape[0]):
        for y in xrange(0, cont_shape[1]):
            if np.abs(content_map[x][y][0] - color[0]) <= 10 and np.abs(content_map[x][y][1] - color[1]) <= 10 and np.abs(content_map[x][y][2] - color[2]) <= 10:
                new_content_img[x][y] = content_img[x][y]
            else:
                new_content_img[x][y] = [124, 117, 104] # The mean used in style transfer
            if np.abs(style_map[x][y][0] - color[0]) <= 10 and np.abs(style_map[x][y][1] - color[1]) <= 10 and np.abs(style_map[x][y][2] - color[2]) <= 10:
                new_style_img[x][y] = style_img[x][y]
            else:
                new_style_img[x][y] = [124, 117, 104] # The mean used in style transfer
    cv2.imwrite(style_output_dir + content_file + '_' + style_file + '_' + str(segment_num) + '.jpg', new_style_img)
    cv2.imwrite(content_output_dir + content_file + '_' + style_file + '_' + str(segment_num) + '.jpg', new_content_img)
    segment_num += 1