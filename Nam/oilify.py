import cv2
import numpy as np
import imutils
np.seterr(over='ignore')

#radius = 4 for iris
#radius = 2 for denali
def my_oilify(image_path, radius=4, intensityLevelSize=10, gap=1):
    img = cv2.imread(image_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = imutils.resize(img, width=256)

    imgHeight, imgWidth = img.shape[0], img.shape[1]

    oilifiedImage = np.zeros(img.shape, np.uint8)

    for i in range(radius, imgHeight - radius, gap):
        for j in range(radius, imgWidth - radius, gap):
            #
            grayLevel = np.zeros((intensityLevelSize, 4))  # Store the number of each gray level
            graySum = [0, 0, 0]  # for the final high frequency gray level mean calculation
            # traversal statistics for small areas
            for m in range(-radius, radius):
                for n in range(-radius, radius):
                    pixlv = int(((img[i + m, j + n, 0] + img[i + m, j + n, 1] + img[i + m, j + n, 2]) / 3) * (
                                intensityLevelSize / 255))
                    if pixlv > 255:
                        pixlv = 255  # Determine pixel level
                    grayLevel[pixlv, 0] += img[i + m, j + n, 0]
                    grayLevel[pixlv, 1] += img[i + m, j + n, 1]
                    grayLevel[pixlv, 2] += img[i + m, j + n, 2]
                    grayLevel[pixlv, 3] += 1  # Calculate the number of corresponding gray levels
                    # Find the highest frequency gray level and its index
            mostLevel = np.max(grayLevel, axis=0)[-1]
            mostLevelIndex = np.argmax(grayLevel, axis=0)[-1]
            # Calculate the mean of all gray values ​​in the highest frequency level

            (b, g, r) = (int(grayLevel[mostLevelIndex, 0] / mostLevel),
                         int(grayLevel[mostLevelIndex, 1] / mostLevel),
                         int(grayLevel[mostLevelIndex, 2] / mostLevel))
            # Write target pixel
            for m in range(gap):
                for n in range(gap):
                    oilifiedImage[i + m, j + n] = (r,g,b)

    return oilifiedImage
