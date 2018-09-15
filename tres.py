import time
import grayscale as gs
import cv2
import numpy as np


# wolne to bardzo :(
def threshold(img, thresh, max_value):
    ts = time.time()
    H, W = img.shape[:2]
    binary = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            binary[i, j] = max_value if img[i, j] > thresh else 0
    t = (time.time() - ts)
    print("Thresh: {:} ms".format(t * 1000))
    return binary


# bardzo szybki threshold 10x razy wolniejszy od modulu
def fun(img, thresh, max_value):
    ts = time.time()
    binary = img
    binary[binary > thresh] = max_value
    binary = (binary > thresh) * binary
    t = (time.time() - ts)
    print("Thresh: {:} ms".format(t * 1000))
    return binary


# wykorzystanie bibliotecznej funkcji najszybsze
def hehe(img, thresh, max_value):
    ts = time.time()
    retval, dst = cv2.threshold(img, thresh, max_value, cv2.THRESH_BINARY)
    t = (time.time() - ts)
    print("Thresh: {:} ms".format(t * 1000))
    return dst


def main():
    img = cv2.imread('hd.jpg')
    gray = gs.grayscale_luna_np(img)
    binary = hehe(gray, 127, 255)
    bin2 = fun(gray, 127, 255)
    # cv2.imshow("jd2", gray)
    # cv2.imshow("thresh", binary)
    # cv2.imshow("jd", bin2)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
