import cv2


def sobel(img):
    '''
    Detects edges using sobel kernel
    '''
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)  # detects horizontal edges
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)  # detects vertical edges
    # combine both edges
    return cv2.bitwise_or(opImgx, opImgy)  # does a bitwise OR of pixel values at each pixel


def my_pencilSketch(image_path):
    # Blur it to remove noise
    img = cv2.imread(image_path, 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # make a negative image
    inverseImage = 255 - img

    # Detect edges from the input image and its negative
    edgImg0 = sobel(img)
    edgImg1 = sobel(inverseImage)
    edgImg = cv2.addWeighted(edgImg0, 1, edgImg1, 1, 0)  # different weights can be tried too

    # Invert the image back
    pencilImage = 255 - edgImg
    pencilImage = cv2.cvtColor(pencilImage, cv2.COLOR_BGR2RGB)
    return pencilImage