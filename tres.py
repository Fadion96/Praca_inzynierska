import time
import grayscale as gs
import cv2
import numpy as np


def otsu(img):
    ts = time.time()
    pixel_number = img.shape[0] * img.shape[1]
    mean_weigth = 1.0 / pixel_number
    his, bins = np.histogram(img, np.arange(0, 257))
    final_thresh = 0
    final_value = 0
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(Wb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(Wf)
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t - 1
            final_value = value
    binary = img.copy()
    binary[img > final_thresh] = 255
    binary[img <= final_thresh] = 0
    t = (time.time() - ts)
    print("Otsu: {:} ms".format(t * 1000))
    return binary


def otsu_2(img):
    ts = time.time()
    pixel_number = img.shape[0] * img.shape[1]
    mean_weigth = 1.0 / pixel_number
    his, bins = np.histogram(img, np.arange(0, 257))
    pcb = 0
    sum_b = 0
    final_thresh = 0
    final_value = 0
    intensity_arr = np.arange(256)
    summ = np.sum(intensity_arr * his)
    for t in bins[:-1]:
        pcb += his[t]
        if pcb == 0:
            continue
        pcf = pixel_number - pcb
        if pcf == 0:
            break
        wb = pcb * mean_weigth
        wf = pcf * mean_weigth

        sum_b += intensity_arr[t] * his[t]
        mub = float(sum_b) / float(wb)
        muf = float(summ - sum_b) / float(wf)
        value = wb * wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    binary = img.copy()
    binary[img > final_thresh] = 255
    binary[img <= final_thresh] = 0
    t = (time.time() - ts)
    print("Otsu_v2: {:} ms".format(t * 1000))
    return binary


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
    binary = img.copy()
    binary[img > thresh] = max_value
    binary[img <= thresh] = 0
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


def hehe2(img):
    ts = time.time()
    ret2, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = (time.time() - ts)
    print("Otsu_cv: {:} ms".format(t * 1000))
    return dst


def main():
    img = cv2.imread('hd.jpg')
    gray = gs.grayscale_luna_np(img)
    bina = hehe(gray, 127, 255)
    bin2 = fun(gray, 127, 255)
    bin3 = otsu(gray)
    bin4 = otsu_2(gray)
    bin5 = hehe2(gray)
    # cv2.imshow("jd", gray)
    # cv2.imshow("jd3", bina)
    # cv2.imshow("jd2", bin2)
    # cv2.imshow("otsu", bin3)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
