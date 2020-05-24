from sklearn.cluster import KMeans
import math
import cv2
import bisect
import scipy.spatial
import numpy as np
import random

def limit_size(img, max_x, max_y=0):
    if max_x == 0: #If max_x is 0, no limit => no change in image
        return img

    if max_y == 0:
        max_y = max_x #If limit for y_plane not set then the image will be limited to max_x

    #If image has same length and height as max_x and max_y (ratio=1)
    #then return image
    #else resize to max_x
    ratio = min(1.0, float(max_x) / img.shape[1], float(max_y) / img.shape[0])

    if ratio != 1.0:
        shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    else:
        return img


def clipped_addition(img, x, _max=255, _min=0):
    if x > 0:
        mask = img > (_max - x)
        img += x
        np.putmask(img, mask, _max)
    if x < 0:
        mask = img < (_min - x)
        img += x
        np.putmask(img, mask, _min)

#Function to add slight changes to palette and increase number of primary colors
def regulate(img, hue=0, saturation=0, luminosity=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    if hue < 0:
        hue = 255 + hue
    hsv[:, :, 0] += hue
    clipped_addition(hsv[:, :, 1], saturation)
    clipped_addition(hsv[:, :, 2], luminosity)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

##Using KNN to get primary colors (palette) in image
def get_palette(img, palette_size, max_img_size=200, n_init=10):
  # scale down the image to speedup kmeans
  img = limit_size(img, max_img_size)

  clt = KMeans(n_clusters=palette_size, n_init=n_init)
  clt.fit(img.reshape(-1, 3))
  pallete = clt.cluster_centers_

  return pallete


#Add more colors to palette
def extend(extensions, palette, base_len):
  extension = [regulate(palette.reshape((1, len(palette), 3)).astype(np.uint8), *hue).reshape((-1, 3)) for hue
              in extensions]

  base_len = base_len if base_len > 0 else len(palette)
  return (np.vstack([palette.reshape((-1, 3))] + extension), base_len)


#Convert palette to image
def to_image(palette, base_len):
  cols = base_len
  rows = int(math.ceil(len(palette) / cols))

  res = np.zeros((rows * 80, cols * 80, 3), dtype=np.uint8)
  for y in range(rows):
    for x in range(cols):
      if y * cols + x < len(palette):
        color = [int(c) for c in palette[y * cols + x]]
        cv2.rectangle(res, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), color, -1)

  return res

def from_gradient(gray):
  fieldx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 15.36
  fieldy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 15.36

  return (fieldx, fieldy)


def smooth(gradientX, gradientY, radius, iterations=1):
  size = 2*radius + 1
  for i in range(iterations):
    new_gradientX = cv2.GaussianBlur(gradientX, (size, size), 0)
    new_gradientY = cv2.GaussianBlur(gradientY, (size, size), 0)

  return (new_gradientX, new_gradientY)


def direction(gradientX, gradientY, i, j):
  return math.atan2(gradientY[i, j], gradientX[i, j])


def magnitude(gradientX, gradientY, i, j):
  return math.hypot(gradientY[i, j], gradientX[i, j])


def randomized_grid(h, w, scale):
    assert (scale > 0)

    r = scale // 2  # Noise for grid

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid


def compute_color_probabilities(pixels, palette, k=9):
    distances = scipy.spatial.distance.cdist(pixels, palette)  # Calculate

    maxima = np.amax(distances, axis=1)  # max of rows

    distances = maxima[:, None] - distances

    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k * len(palette) * distances)
    summ = np.sum(distances, 1)  # sum by rows
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)


def color_select(probabilities, palette):
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]


def my_pointilism(image_path):
    img = cv2.imread(image_path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    palette = get_palette(img, palette_size=20)

    base_len = len(palette)
    new_palette, new_base_len = extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)], palette, base_len)

    gradientX, gradientY = from_gradient(gray)
    gradientX, gradientY = smooth(gradientX, gradientY, radius=0)

    res = cv2.medianBlur(img, 11)
    # define a randomized grid of locations for the brush strokes
    grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
    batch_size = 10000

    stroke_scale = int(math.ceil(max(img.shape) / 1000))

    for h in range(0, len(grid), batch_size):
        # get the pixel colors at each point of the grid
        pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
        # precompute the probabilities for each color in the palette
        # lower values of k means more randomnes
        color_probabilities = compute_color_probabilities(pixels, new_palette, k=9)

        for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
            color = color_select(color_probabilities[i], new_palette)
            angle = math.degrees(direction(gradientX, gradientY, y, x)) + 90
            length = int(round(stroke_scale + stroke_scale * math.sqrt(magnitude(gradientX, gradientY, y, x))))

            # draw the brush stroke
            cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

    return limit_size(res, 1080)