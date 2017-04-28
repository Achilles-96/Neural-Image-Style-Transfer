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
    current_content_map = content_map[:,:,0].copy()
    current_style_map = style_map[:,:,0].copy()

    for x in range(0, cont_shape[0]):
        for y in range(0, cont_shape[1]):
            if np.abs(content_map[x][y][0] - color[0]) <= 10 and np.abs(content_map[x][y][1] - color[1]) <= 10 and np.abs(content_map[x][y][2] - color[2]) <= 10:
                current_content_map[x][y] = 255.0
            else:
                current_content_map[x][y] = 0.0
            if np.abs(style_map[x][y][0] - color[0]) <= 10 and np.abs(style_map[x][y][1] - color[1]) <= 10 and np.abs(style_map[x][y][2] - color[2]) <= 10:
                current_style_map[x][y] = 255.0
            else:
                current_style_map[x][y] = 0.0

    kernel = np.ones((9,9), np.float32) / 81
    current_content_map = cv2.dilate(current_content_map, kernel)
    current_style_map = cv2.dilate(current_style_map, kernel)
    current_content_map = cv2.filter2D(current_content_map, -1, kernel)
    current_style_map = cv2.filter2D(current_style_map, -1, kernel)

    new_content_img = content_img.copy()
    new_style_img = style_img.copy()
    for x in xrange(0, cont_shape[0]):
        for y in xrange(0, cont_shape[1]):
            new_content_img[x][y][0] = content_img[x][y][0]*(current_content_map[x][y]/255.0) + 124*(1.0 - current_content_map[x][y]/255.0)
            new_content_img[x][y][1] = content_img[x][y][1]*(current_content_map[x][y]/255.0) + 117*(1.0 - current_content_map[x][y]/255.0)
            new_content_img[x][y][2] = content_img[x][y][2]*(current_content_map[x][y]/255.0) + 104*(1.0 - current_content_map[x][y]/255.0)
            new_style_img[x][y][0] = style_img[x][y][0]*(current_style_map[x][y]/255.0) + 124*(1.0 - current_style_map[x][y]/255.0)
            new_style_img[x][y][1] = style_img[x][y][1]*(current_style_map[x][y]/255.0) + 117*(1.0 - current_style_map[x][y]/255.0)
            new_style_img[x][y][2] = style_img[x][y][2]*(current_style_map[x][y]/255.0) + 104*(1.0 - current_style_map[x][y]/255.0)
 
    cv2.imwrite(style_output_dir + content_file + '_' + style_file + '_' + str(segment_num) + '.jpg', new_style_img)
    cv2.imwrite(content_output_dir + content_file + '_' + style_file + '_' + str(segment_num) + '.jpg', new_content_img)
    segment_num += 1