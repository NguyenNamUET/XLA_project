import cv2
import numpy as np


def scale_to_0_255(image):
    min_val = np.min(image)
    max_val = np.max(image)
    new_image = (image - min_val) / (max_val - min_val) # 0-1
    new_image *= 255
    return new_image


def my_canny(img_path, min_val, max_val, sobel_size=3, is_L2_gradient=False):
    img = cv2.imread(img_path, 0)
     # giảm nhiễu ảnh
    smooth_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)       

    # tính cường độ và hướng của Gradient  
    Gradient_x = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=sobel_size)      
    Gradient_y = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    if is_L2_gradient:
        edge_gradient = np.sqrt(Gradient_x*Gradient_x + Gradient_y*Gradient_y)
    else:
        edge_gradient = np.abs(Gradient_x) + np.abs(Gradient_y)

    Gradient_angle = np.arctan2(Gradient_y, Gradient_x) * 180 / np.pi           
    
    # gom về 4 hướng 0 độ, 45 độ, 90 độ, 135 độ
    Gradient_angle = np.abs(Gradient_angle)                                     
    Gradient_angle[Gradient_angle <= 22.5] = 0
    Gradient_angle[Gradient_angle >= 157.5] = 0
    Gradient_angle[(Gradient_angle > 22.5) * (Gradient_angle < 67.5)] = 45
    Gradient_angle[(Gradient_angle >= 67.5) * (Gradient_angle <= 112.5)] = 90
    Gradient_angle[(Gradient_angle > 112.5) * (Gradient_angle <= 157.5)] = 135
    
    # Loại bỏ các pixel ở vị trí không phải cực đại toàn cục (sử dụng 1 filter 3x3)
    keep_mask = np.zeros(smooth_img.shape, np.uint8)
    for y in range(1, edge_gradient.shape[0]-1):
        for x in range(1, edge_gradient.shape[1]-1):
            area_grad_intensity = edge_gradient[y-1:y+2, x-1:x+2]   # 3x3 filter
            area_angle = Gradient_angle[y-1:y+2, x-1:x+2]           # 3x3 filter
            current_angle = area_angle[1,1]
            current_grad_intensity = area_grad_intensity[1,1]
            
            # so sánh pixel trung tâm với 2 pixel lân cận theo hướng Gradient
            if current_angle == 0:
                if current_grad_intensity > max(area_grad_intensity[1,0], area_grad_intensity[1,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 45:
                if current_grad_intensity > max(area_grad_intensity[2,0], area_grad_intensity[0,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 90:
                if current_grad_intensity > max(area_grad_intensity[0,1], area_grad_intensity[2,1]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 135:
                if current_grad_intensity > max(area_grad_intensity[0,0], area_grad_intensity[2,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
    
    # Lọc Thresholding    
    canny_mask = np.zeros(smooth_img.shape, np.uint8)
    canny_mask[(keep_mask>0) * (edge_gradient>min_val)] = 255
    
    return scale_to_0_255(canny_mask)
