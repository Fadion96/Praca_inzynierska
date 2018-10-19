import time
import grayscale as gs
import tres as tr
import cv2
import numpy as np


# TODO: Przyspieszyc to diametralnie(zmiana koncepcji czy tez watki) złe podeście do poprawki
def dilate(img, structuring_element, anchor):
    ts = time.time()
    h, w = img.shape
    result = img.copy()
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(h):
        for j in range(w):
            if not bool(img[i][j]):
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
                result[i][j] = np.max(window[np.where(struct_window == 255)])

    t = (time.time() - ts)
    print("Dilate: {:} ms".format(t * 1000))
    return result


# TODO: Przyspieszyc to diametralnie(zmiana koncepcji czy tez watki)
def erode(img, structuring_element, anchor):
    h, w = img.shape
    result = img.copy()
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(h):
        for j in range(w):
            x_start = j - anchor[1]
            x_end = j + distance_x
            y_start = i - anchor[0]
            y_end = i + distance_y
            struct_x_start = 0
            struct_x_end = structuring_element.shape[1]
            struct_y_start = 0
            struct_y_end = structuring_element.shape[0]
            if x_start < 0 or  x_end > w or y_start < 0 or y_end > h:
                result[i][j] = 0
            else:
                struct_window = structuring_element[struct_y_start:struct_y_end,
                                struct_x_start:struct_x_end]
                window = img[y_start:y_end, x_start:x_end] & struct_window
                tmp = window[np.where(struct_window == 255)]
                result[i][j] = np.min(tmp)
    return result


# wolniejsze od ifow
def erode2(img, structuring_element, anchor):
    h, w = img.shape
    result = img.copy()
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(h):
        for j in range(w):
            x_start = max(0, j - anchor[1])
            x_end = min(w, j + distance_x)
            y_start = max(0, i - anchor[0])
            y_end = min(h, i + distance_y)
            struct_x_start = max(0, -(j - anchor[1]))
            struct_x_end = min(structuring_element.shape[1], structuring_element.shape[1] - (j + distance_x) + w)
            struct_y_start = max(0, -(i - anchor[0]))
            struct_y_end = min(structuring_element.shape[0], structuring_element.shape[0] - (i + distance_y) + h)
            struct_window = structuring_element[struct_y_start:struct_y_end, struct_x_start:struct_x_end]
            window = img[y_start:y_end, x_start:x_end] & struct_window
            result[i][j] = np.min(window[np.where(struct_window == 255)])
    return result


def opening(img, structuring_element, anchor):
    return dilate(erode(img, structuring_element, anchor), structuring_element, anchor)


def closing(img, structuring_element, anchor):
    return erode(dilate(img, structuring_element, anchor), structuring_element, anchor)


def hit_miss(img, structuring_element, anchor):
    hit_element = structuring_element.copy()
    miss_element = structuring_element.copy()
    hit_element[structuring_element == 255] = 255
    hit_element[structuring_element != 255] = 0
    miss_element[structuring_element == -255] = 255
    miss_element[structuring_element != -255] = 0
    hit_img = erode(img, hit_element, anchor)
    miss_img = erode(invert(img), miss_element, anchor)
    return hit_img & miss_img


def invert(img):
    result = img.copy()
    result[img == 255] = 0
    result[img == 0] = 255
    return result


def main():
    img = cv2.imread('template.jpg')
    gray = gs.grayscale_luna_np(img)
    binary = tr.otsu_2(gray)
    element = np.array([[0, 255, 0], [-255, 255, 255], [-255, -255, 0]])

    hit = hit_miss(binary, element, (1, 1))
    cv2.imshow("first", binary)
    cv2.imshow("hit_miss", hit)
    cv2.waitKey()
    # size = 1
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * size + 1, 2 * size + 1))
    # element2 = element * 255


if __name__ == '__main__':
    main()
