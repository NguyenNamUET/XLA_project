import cv2
import numpy as np
import math


def my_waving(image_path, waving_direction=0):
    origin_image = cv2.imread(image_path)
    resize_image = cv2.resize(origin_image,(400,600))
    rows, cols = resize_image.shape[:2]

    # Sóng dọc
    if waving_direction == 0:
        vertical_image = np.zeros(resize_image.shape, dtype=resize_image.dtype)

        for x in range(rows):
            for y in range(cols):
                offset_x = int(25.0 * math.sin(2 * 3.14 * x / 180))
                offset_y = 0
                if y+offset_x < rows:
                    vertical_image[x,y] = resize_image[x,(y+offset_x)%cols]
                else:
                    vertical_image[x,y] = 0

        return vertical_image

    # sóng ngang
    elif waving_direction == 1:
        horizontal_image = np.zeros(resize_image.shape, dtype=resize_image.dtype)

        for i in range(rows):
            for j in range(cols):
                offset_x = 0
                offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
                if i+offset_y < rows:
                    horizontal_image[i,j] = resize_image[(i+offset_y)%rows,j]
                else:
                    horizontal_image[i,j] = 0

        return horizontal_image

    # sóng 2 hướng
    elif waving_direction == 2:
        waving_filter = np.zeros(resize_image.shape, dtype=resize_image.dtype)

        for i in range(rows):
            for j in range(cols):
                offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
                if i+offset_y < rows and j+offset_x < cols:
                    waving_filter[i,j] = resize_image[(i+offset_y)%rows,(j+offset_x)%cols]
                else:
                    waving_filter[i,j] = 0

        return waving_filter