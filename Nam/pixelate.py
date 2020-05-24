import cv2

def my_pixelate(image_path, pixelX = 32, pixelY=32):
    input = cv2.imread(image_path)

    # Get input size
    height, width = input.shape[:2]

    # Desired "pixelated" size
    w, h = (pixelX, pixelY)

    # Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    return output