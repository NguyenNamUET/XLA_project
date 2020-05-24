import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def my_vignette(image_path):
    origin_image = cv2.imread(image_path)
    rows, cols = origin_image.shape[:2]

    # khởi tạo vignette mask sử dụng GaussianKernel
    x_kernel = cv2.getGaussianKernel(rows, 200)             # khởi tạo GaussianKernel size rows
    y_kernel = cv2.getGaussianKernel(cols, 200)             # khởi tạo GaussianKernel size cols
    kernel = x_kernel * y_kernel.T
    mask = kernel/kernel.max()
    vignette = np.copy(origin_image)

    # áp dụng từng mask cho từng channel của ảnh
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask

    return vignette