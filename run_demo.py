from Nam.pixelate import *
import matplotlib.pyplot as plt

import cv2
if __name__ == '__main__':
    image_path = '/home/nguyennam/PycharmProjects/XLA_project/images/iris.jpg'
    origin_img = cv2.imread(image_path)
    canny = my_pixelate(image_path)

    cv2.imshow('origin_image', origin_img)
    cv2.imshow('canny_filter', canny)
    cv2.waitKey()
    cv2.destroyAllWindows()