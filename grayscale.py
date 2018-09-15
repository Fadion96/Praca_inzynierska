import cv2
import numpy as np
import time


# kazdy kolor 1/3 robione manualnie (wolne)
def grayscale_third_manual(img):
    ts = time.time()
    H, W = img.shape[:2]
    gray = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            gray[i, j] = np.clip(0.33 * img[i, j, 0] + 0.33 * img[i, j, 1] + 0.33 * img[i, j, 2], 0, 255)
    t = (time.time() - ts)
    print("Loop: {:} ms".format(t * 1000))
    return gray


# kazdy kolor 1/3 robione z numpy (szybkie)
def grayscale_third_np(img):
    ts = time.time()
    w = np.array([[[0.33, 0.33, 0.33]]])
    gray = cv2.convertScaleAbs(np.sum(img * w, axis=2))
    t = (time.time() - ts)
    print("Loop: {:} ms".format(t * 1000))
    return gray


# kolory odzwierciedlane przez ludzkie oko w standardzie hd(?) manual wolny
def grayscale_luna_manual(img):
    ts = time.time()
    H, W = img.shape[:2]
    gray = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            gray[i, j] = np.clip(0.07 * img[i, j, 0] + 0.72 * img[i, j, 1] + 0.21 * img[i, j, 2], 0, 255)
    t = (time.time() - ts)
    print("Loop: {:} ms".format(t * 1000))
    return gray


# kolory odzwierciedlane przez ludzkie oko w standardzie hd(?) numpy szybkie
def grayscale_luna_np(img):
    ts = time.time()
    w = np.array([[[0.07, 0.72, 0.21]]])
    gray = cv2.convertScaleAbs(np.sum(img * w, axis=2))
    t = (time.time() - ts)
    print("Loop: {:} ms".format(t * 1000))
    return gray


def main():
    img = cv2.imread('hd.jpg')
    # gray = grayscale_third_manual(img)
    # gray2 = grayscale_third_np(img)
    # gray3 = grayscale_luna_manual(img)
    gray4 = grayscale_luna_np(img)
    # cv2.imshow("original", img)
    # cv2.imshow("gray", gray)
    # cv2.imshow("gray2", gray2)
    # cv2.imshow("gray3", gray3)
    cv2.imshow("gray4", gray4)
    cv2.waitKey()


if __name__ == '__main__':
    main()
