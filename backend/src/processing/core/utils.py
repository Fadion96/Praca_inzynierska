import base64

import numpy as np
import cv2
import time
from skimage.exposure import rescale_intensity


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("{:} ms".format((te - ts) * 1000))

        return result

    return timed


def to_image_string(image_filepath):
    with open(image_filepath, 'rb') as image:
        encoded_string = base64.b64encode(image.read())
        return encoded_string


def from_base64(base64_data):
    nparr = np.fromstring(base64.b64decode(base64_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def invert_channel(img):
    result = img.copy()
    result = 255 - result
    return result


def histogram_equalization_channel(img):
    result = img.copy()
    his, bins = np.histogram(img, np.arange(0, 257))
    cumulative_distribution = his.cumsum()
    lut = np.uint8(255 * cumulative_distribution / cumulative_distribution[-1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = lut[img[i, j]]
    return result


def make_lut_stretching(minimum, maximum):
    lut = [0] * 256
    for i in range(256):
        lut[i] = int((i - minimum) / (maximum - minimum) * 255)
    return lut


def make_lut_contrast(alpha):
    lut = np.array([0] * 256)
    for i in range(256):
        lut[i] = int(alpha * (i - 127) + 127)
    lut[lut > 255] = 255
    lut[lut < 0] = 0
    return lut


def make_lut_erosion(element):
    lut_size = 2 ** element.size
    lut = np.array([0] * lut_size)
    number = 0
    for bit in element.flatten():
        number = (number << 1) | bit
    lut[number] = 255
    return lut


def convolution_channel(img, kernel):
    height, width = img.shape
    kernel_height, kernel_width = kernel.shape
    divisor = np.sum(kernel)
    if divisor == 0:
        divisor = 1
    padding = (kernel_height - 1) // 2
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    result = np.zeros(img.shape, dtype="float16")
    for y in range(padding, height + padding):
        for x in range(padding, width + padding):
            window = padded_img[y - padding: y + padding + 1, x - padding: x + padding + 1]
            value = np.sum(window * kernel) // divisor
            result[y - padding, x - padding] = value
    result = rescale_intensity(result, in_range=(0, 255), out_range='uint8').astype("uint8")
    return result


def multiplication_channel(img, mask):
    result = img & mask
    return result


def bilinear_interp(img, x, y):
    height, width = img.shape[:2]
    x0 = np.floor(x).astype(int)
    x1 = np.floor(x).astype(int) + 1
    y0 = np.floor(y).astype(int)
    y1 = np.floor(y).astype(int) + 1

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    a = img[y0, x0]
    b = img[y1, x0]
    c = img[y0, x1]
    d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return a * wa + b * wb + c * wc + d * wd


def bilinear_channel(img, x, y):
    height, width = img.shape[:2]
    result = np.zeros((y, x), dtype="uint8")
    height_ratio = height / y
    width_ratio = width / x

    for i in range(y):
        for j in range(x):
            result[i, j] = bilinear_interp(img, j * width_ratio, i * height_ratio)
    return result
