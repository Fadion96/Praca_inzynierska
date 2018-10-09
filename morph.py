import time
import grayscale as gs
import tres as tr
import cv2
import numpy as np


# TODO: Przyspieszyc to diametralnie(zmiana koncepcji czy tez watki)
def dilate(img, structuring_element, anchor):
    ts = time.time()
    h, w = img.shape
    result = img.copy()
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(h):
        for j in range(w):
            if bool(img[i][j]):
                x_start = j - anchor[1]
                x_end = j + distance_x
                y_start = i - anchor[0]
                y_end = i + distance_y
                struct_x_start = 0
                struct_x_end = structuring_element.shape[1]
                struct_y_start = 0
                struct_y_end = structuring_element.shape[0]
                if x_start < 0:
                    struct_x_start -= x_start
                    x_start = 0
                if x_end > w:
                    struct_x_end = struct_x_end - (x_end - w)
                    x_end = w
                if y_start < 0:
                    struct_y_start -= y_start
                    y_start = 0
                if y_end > h:
                    struct_y_end = struct_y_end - (y_end - h)
                    y_end = h
                window = img[y_start:y_end, x_start:x_end] \
                         | structuring_element[struct_y_start:struct_y_end,
                           struct_x_start:struct_x_end]
                result[y_start:y_end, x_start:x_end] = window

    t = (time.time() - ts)
    print("Dilate: {:} ms".format(t * 1000))
    return result


# TODO: Przyspieszyc to diametralnie(zmiana koncepcji czy tez watki)
def erode(img, structuring_element, anchor):
    ts = time.time()
    h, w = img.shape
    result = img.copy()
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(h):
        for j in range(w):
            if bool(img[i][j]):
                x_start = j - anchor[1]
                x_end = j + distance_x
                y_start = i - anchor[0]
                y_end = i + distance_y
                struct_x_start = 0
                struct_x_end = structuring_element.shape[1]
                struct_y_start = 0
                struct_y_end = structuring_element.shape[0]
                if x_start < 0:
                    struct_x_start -= x_start
                    x_start = 0
                if x_end > w:
                    struct_x_end = struct_x_end - (x_end - w)
                    x_end = w
                if y_start < 0:
                    struct_y_start -= y_start
                    y_start = 0
                if y_end > h:
                    struct_y_end = struct_y_end - (y_end - h)
                    y_end = h
                struct_window = structuring_element[struct_y_start:struct_y_end,
                                struct_x_start:struct_x_end]
                window = img[y_start:y_end, x_start:x_end] \
                         & struct_window
                result[i][j] = np.min(window[np.where(struct_window == 255)])
    t = (time.time() - ts)
    print("Erode: {:} ms".format(t * 1000))
    return result


def main():
    img = cv2.imread('hd.jpg')
    gray = gs.grayscale_luna_np(img)
    binary = tr.otsu_2(gray)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    element = element * 255
    # element = np.array(3 * [3 * [255]])
    dilatation_dst = dilate(binary, element, (1, 1))
    erosion_dst = erode(binary, element, (1, 1))
    cv2.imshow("binary", binary)
    # cv2.imshow("dilate", dilatation_dst)
    cv2.imshow("erode", erosion_dst)
    cv2.waitKey()


if __name__ == '__main__':
    main()
