import numpy as np
import cv2
import time
import multiprocessing
from skimage.exposure import rescale_intensity


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def load_image(path) -> np.ndarray:
    img = cv2.imread(path)
    if isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Cannot load image")


def grayscale_luma(img):
    """
    Funkcja, która przekształca obraz w modelu RGB (przechowywany jako BGR) na skalę szarości
    w standardzie ITU-R BT.709 dla HDTV.
    :param img: Tensor o wymiarach M x N x 3 przedstawiający liczbową reprezentacje obrazu w modelu RGB
    :return: Macierz o wymiarach M x N przedstawiajacy liczbową reprezentacje obrazu w skali szarości
    """
    w = np.array([[[0.07, 0.72, 0.21]]])
    gray = cv2.convertScaleAbs(np.sum(img * w, axis=2))
    return gray


def grayscale(img, red_weight, green_weight, blue_weight):
    """
    Funkcja, która przekształca obraz w modelu RGB (przechowywany jako BGR) na skalę szarości
    wedle podanych wag dla kolorów przez użytkownika.
    :param img: Tensor o wymiarach M x N x 3 przedstawiający liczbowa reprezentacje obrazu w modelu RGB.
    :param red_weight: Waga dla koloru czerwonego.
    :param green_weight: Waga dla koloru zielonego.
    :param blue_weight: Waga dla koloru niebieskiego.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu w skali szarości.
    """
    w = np.array([[[blue_weight, green_weight, red_weight]]])
    gray = cv2.convertScaleAbs(np.sum(img * w, axis=2))
    return gray


def otsu(img):
    """
    Fukncja, która przekształca obraz w skali szarości na obraz binarny, za pomoca metody Otsu.
    :param img: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu w skali szarości.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    """
    pixel_number = img.shape[0] * img.shape[1]
    mean_weight = 1.0 / pixel_number
    his, bins = np.histogram(img, np.arange(0, 257))
    pixel_count_background = 0
    sum_background = 0
    final_threshold = 0
    final_value = 0
    intensity_array = np.arange(256)
    sum_image = np.sum(intensity_array * his)
    for threshold in bins[:-1]:
        pixel_count_background += his[threshold]
        if pixel_count_background == 0:
            continue
        pixel_count_foreground = pixel_number - pixel_count_background
        if pixel_count_foreground == 0:
            break
        weight_background = pixel_count_background * mean_weight
        weight_foreground = pixel_count_foreground * mean_weight

        sum_background += intensity_array[threshold] * his[threshold]
        mean_background = float(sum_background) / float(weight_background)
        mean_foreground = float(sum_image - sum_background) / float(weight_foreground)
        value = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if value > final_value:
            final_threshold = threshold
            final_value = value
    binary = img.copy()
    binary[img > final_threshold] = 255
    binary[img <= final_threshold] = 0
    return binary


def standard_threshold(img, threshold):
    """
    Fukncja, która przekształca obraz w skali szarości na obraz binarny z podaniem progu przez użytkownika.
    :param img: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu w skali szarości.
    :param threshold: Ustalony przez użytkownika próg.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    """
    binary = img.copy()
    binary[img > threshold] = 255
    binary[img <= threshold] = 0
    return binary


def dilation(img, structuring_element, anchor):
    """
    Fukcja, która na obrazie binarnym wykonuje morfologiczną operację dylacji.
    :param img: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest dylacja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana dylacja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej dylacji.
    """
    height, width = img.shape
    result = img.copy()
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(height):
        for j in range(width):
            if not bool(img[i, j]):
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
                if x_end > width:
                    struct_x_end = struct_x_end - (x_end - width)
                    x_end = width
                if y_start < 0:
                    struct_y_start -= y_start
                    y_start = 0
                if y_end > height:
                    struct_y_end = struct_y_end - (y_end - height)
                    y_end = height
                struct_window = structuring_element[struct_y_start:struct_y_end,
                                struct_x_start:struct_x_end]
                window = img[y_start:y_end, x_start:x_end] \
                         & struct_window
                result[i, j] = np.max(window[np.where(struct_window == 1)])

    return result


def erosion(img, structuring_element, anchor):
    """
    Fukcja, która na obrazie binarnym wykonuje morfologiczną operację erozji.
    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
    """
    height, width = img.shape
    result = img.copy()

    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(height):
        for j in range(width):
            x_start = j - anchor[1]
            x_end = j + distance_x
            y_start = i - anchor[0]
            y_end = i + distance_y
            struct_x_start = 0
            struct_x_end = structuring_element.shape[1]
            struct_y_start = 0
            struct_y_end = structuring_element.shape[0]
            if x_start < 0 or x_end > width or y_start < 0 or y_end > height:
                result[i, j] = 0
            else:
                struct_window = structuring_element[struct_y_start:struct_y_end,
                                struct_x_start:struct_x_end]
                window = img[y_start:y_end, x_start:x_end] & struct_window
                result[i, j] = np.min(window[np.where(struct_window == 1)]) * 255
    return result


def bin_erosion(img, structuring_element, anchor):
    """
    Fukcja, która na obrazie binarnym wykonuje morfologiczną operację erozji.
    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
    """
    height, width = img.shape
    result = img.copy()
    lut = make_lut_erosion(structuring_element)
    distance_y = structuring_element.shape[0] - anchor[0]
    distance_x = structuring_element.shape[1] - anchor[1]
    for i in range(height):
        for j in range(width):
            x_start = j - anchor[1]
            x_end = j + distance_x
            y_start = i - anchor[0]
            y_end = i + distance_y
            struct_x_start = 0
            struct_x_end = structuring_element.shape[1]
            struct_y_start = 0
            struct_y_end = structuring_element.shape[0]
            if x_start < 0 or x_end > width or y_start < 0 or y_end > height:
                result[i, j] = 0
            else:
                struct_window = structuring_element[struct_y_start:struct_y_end,
                                struct_x_start:struct_x_end]
                window = img[y_start:y_end, x_start:x_end] & struct_window
                number = 0
                for bit in window.flatten():
                    number = (number << 1) | bit
                result[i, j] = lut[number]
    return result


def opening(img, structuring_element, anchor):
    """
    Fukcja, która na obrazie binarnym wykonuje morfologiczną operację otwarcia, czyli wykonania operacji erozji, a nastepnie dylacji na obrazie.
    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywane jest otwarcie.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywane otwarcie.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanym otwarciu.
    """
    return dilation(erosion(img, structuring_element, anchor), structuring_element, anchor)


def closing(img, structuring_element, anchor):
    """
    Fukcja, która na obrazie binarnym wykonuje morfologiczną operację otwarcia, czyli wykonania operacji dylacji, a nastepnie erozji na obrazie.
    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywane jest domkniecie.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywane domkniecie.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanym domknieciu.
    """
    return erosion(dilation(img, structuring_element, anchor), structuring_element, anchor)


def hit_miss(img, structuring_element, anchor):
    """
    Fukcja, która na obrazie binarnym wykonuje morfologiczną operację Hit-or-miss, , którą to można przedstawić jako część wspólną erozji
    erozji obrazu z elementem strukturalnym oznaczającym "trafienia" oraz erozji odwróconego obrazu z elementem strukturalnym oznaczającym "pudła"
    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
    """
    hit_element = structuring_element.copy()
    miss_element = structuring_element.copy()
    hit_element[structuring_element == 1] = 1
    hit_element[structuring_element != 1] = 0
    miss_element[structuring_element == -1] = 1
    miss_element[structuring_element != -1] = 0
    hit_img = erosion(img, hit_element, anchor)
    miss_img = erosion(invert_channel(img), miss_element, anchor)
    return hit_img & miss_img


def morph_gradient(img, structuring_element, anchor):
    """

    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
    """
    dilation_dst = dilation(img, structuring_element, anchor)
    erosion_dst = erosion(img, structuring_element, anchor)
    return dilation_dst - erosion_dst


def top_hat(img, structuring_element, anchor):
    """

    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
    """
    opening_dst = opening(img, structuring_element, anchor)
    return img - opening_dst


def black_hat(img, structuring_element, anchor):
    """

    :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
    :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
    :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
    :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
    """
    closing_dst = closing(img, structuring_element, anchor)
    return closing_dst - img


def invert_channel(img):
    result = img.copy()
    result = 255 - result
    return result


def invert(img):
    ts = time.time()
    if len(img.shape) == 3:
        channels = []
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        channels.append(invert_channel(b))
        channels.append(invert_channel(g))
        channels.append(invert_channel(r))
        result = cv2.merge(channels)
        t = (time.time() - ts)
        print("RBG negation {:} ms".format(t * 1000))
    else:
        result = invert_channel(img)
        t = (time.time() - ts)
        print("Grayscale negation {:} ms".format(t * 1000))

    return result


def change_brightness(img, value):
    height, width = img.shape
    result = img.copy()
    for i in range(height):
        for j in range(width):
            brightness = result[i, j] + value
            if brightness > 255:
                result[i, j] = 255
            elif brightness < 0:
                result[i, j] = 0
            else:
                result[i, j] = brightness
    return result


def change_contrast(img, alpha):
    height, width = img.shape
    result = img.copy()
    lut = make_lut_contrast(alpha)
    for i in range(height):
        for j in range(width):
            result[i, j] = lut[img[i, j]]
    return result


def stretching_histogram(img):
    height, width = img.shape
    result = img.copy()
    his, bins = np.histogram(img, np.arange(0, 257))
    non_zero = [index for index, item in enumerate(his) if item != 0]
    minimum = non_zero[0]
    maximum = non_zero[-1]
    lut = make_lut_stretching(minimum, maximum)
    for i in range(height):
        for j in range(width):
            result[i, j] = lut[img[i, j]]
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
    result = np.zeros(img.shape, dtype="float32")
    for y in range(padding, height + padding):
        for x in range(padding, width + padding):
            window = padded_img[y - padding: y + padding + 1, x - padding: x + padding + 1]
            value = np.sum(window * kernel) // divisor
            result[y - padding, x - padding] = value
    result = rescale_intensity(result, in_range=(0, 255))
    result = (result * 255).astype("uint8")
    return result


def multiplication_channel(img, mask):
    result = img & mask
    return result


def multiplication(img, mask):
    ts = time.time()
    if img.shape == mask.shape:
        result = multiplication_channel(img, mask)
        t = (time.time() - ts)
        print("Grayscale multi {:} ms".format(t * 1000))
    else:
        channels = []
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        channels.append(multiplication_channel(b, mask))
        channels.append(multiplication_channel(g, mask))
        channels.append(multiplication_channel(r, mask))
        result = cv2.merge(channels)
        t = (time.time() - ts)
        print("RGB multi {:} ms".format(t * 1000))

    return result


def convolution(img, kernel):
    ts = time.time()
    if len(img.shape) == 3:
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        pool = multiprocessing.Pool(processes=3)
        channels = pool.starmap(convolution_channel, [(b, kernel), (g, kernel), (r, kernel)])
        result = cv2.merge(channels)
        t = (time.time() - ts)
        print("RBG convolution {:} ms".format(t * 1000))
    else:
        result = convolution_channel(img, kernel)
        t = (time.time() - ts)
        print("Grayscale convolution {:} ms".format(t * 1000))

    return result


def histogram_equalization(img):
    ts = time.time()
    if len(img.shape) == 3:
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        pool = multiprocessing.Pool(processes=3)
        channels = pool.map(histogram_equalization_channel, [b, g, r])
        result = cv2.merge(channels)
        t = (time.time() - ts)
        print("RBG HE {:} ms".format(t * 1000))
    else:
        result = histogram_equalization_channel(img, )
        t = (time.time() - ts)
        print("Grayscale HE {:} ms".format(t * 1000))

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
            result[i, j] = bilinear_interp(img, j*width_ratio, i*height_ratio)
    return result


def bilinear(img, x, y):
    ts = time.time()
    if len(img.shape) == 3:
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        pool = multiprocessing.Pool(processes=3)
        channels = pool.starmap(bilinear_channel, [(b, x, y), (g, x, y), (r, x, y),])
        result = cv2.merge(channels)
        t = (time.time() - ts)
        print("RBG bilinear {:} ms".format(t * 1000))
    else:
        result = bilinear_channel(img, x, y)
        t = (time.time() - ts)
        print("Grayscale bilinear {:} ms".format(t * 1000))
    return result


# wstep - cel pracy, dlaczego problem jest interesujacy, jakies porowanie z istniejacymi rozwiazaniami
# wymagania fukcjonalne
# algorytmy, wykorzystane narzedzia itd
# opis implementacji
# instrukcja, instalacja obsluga
# testy poprawnosci (automatyczne)
# user experience, dokumentacja, ergonomia, łatwość użytkowania
# co warto dodac, algorytmy, czy funkcjonalnosc
# komentarz do nich
# podsumowanie, co mozna rozwinac (jeden czy dwa pomysły bardziej opisać)


# rescaling
# do gui rozwidlenie zlaczenie
# weryfikacja typow (klasa)
# semafor na czekanie sciezek
#

def main():
    img = load_image('lena.tif')
    gray = grayscale_luma(img)
    test = bilinear(img, 256, 256)

    # contgray = change_contrast(gray, 0.7)
    # neg = invert(contgray)
    # # mask_grey = grayscale_luma(mask)
    # # bin_mask = otsu(mask_grey)
    # # multi = image_multiplication(img, mask)
    # sharpen = np.array((
    #     [0, -1, 0],
    #     [-1, 5, -1],
    #     [0, -1, 0]), dtype="int")
    # conv = convolution(img, sharpen)
    # conv2 = convolution(gray, sharpen)
    # he = histogram_equalization(conv)
    # he2 = histogram_equalization(conv2)
    # neg = invert(img)
    # neg2 = invert(gray)
    # laplacian = np.array((
    #     [0, 1, 0],
    #     [1, -4, 1],
    #     [0, 1, 0]), dtype="int")
    # lap = convolution(img, laplacian)
    # lap2 = convolution(gray, laplacian)
    # bin_lap = otsu(lap2)
    # mask = multiplication(img, lap)
    # mask2 = multiplication(img, bin_lap)
    # conv2 = convolution(gray, laplacian)
    # lap_bin = otsu(conv2)
    # test = image_multiplication(neg, lap_bin)
    # bt = image_multiplication(beq, test)
    # gt = image_multiplication(geq, test)
    # rt = image_multiplication(req, test)
    # test2 = cv2.merge((bt, gt, rt))
    # test3 = cv2.merge((invert(bt), invert(gt), invert(rt)))
    # test4 = cv2.merge((histogram_equalization(invert(bt)), histogram_equalization(invert(gt)), histogram_equalization(invert(rt))))
    # cv2.imshow("img", img)
    # cv2.imshow("gray", gray)
    # cv2.imshow("lap", lap)
    # cv2.imshow("lap2", lap2)
    # cv2.imshow("bin_lap", bin_lap)
    # cv2.imshow("mask", mask)
    # cv2.imshow("mask2", mask2)
    cv2.imshow("test", test)
    cv2.waitKey()
    # cv2.imwrite("img.jpg", img)
    # cv2.imwrite("gray.jpg", gray)
    # cv2.imwrite("lap.jpg", lap)
    # cv2.imwrite("lap2.jpg", lap2)
    # cv2.imwrite("bin_lap.jpg", bin_lap)
    # cv2.imwrite("mask.jpg", mask)
    # cv2.imwrite("mask2.jpg", mask2)
    # cv2.imshow("conv", conv)
    # cv2.imshow("conv2", conv2)
    # cv2.imshow("he", he)
    # cv2.imshow("he2", he2)
    # cv2.imshow("neg", neg)
    # cv2.imshow("neg2", neg2)
    # cv2.imshow("cont", contgray)
    # cv2.imshow("lap", conv2)
    # cv2.imshow("binlap", lap_bin)
    # cv2.imshow("neg", neg)
    # cv2.imshow("eq", eq)
    # cv2.imshow("test2", test2)
    # cv2.imshow("test3", test3)
    # cv2.imshow("end", test4)
    # binary = otsu(gray)
    # element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # size = 1
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * size + 1, 2 * size + 1))
    # ero = erosion(binary, element, (size, size))
    # ero2 = bin_erosion(binary, element, (size, size))

    # wrapped = wrapper(erosion, binary, element, (1, 1))
    # wrapped2 = wrapper(bin_erosion, binary, element, (1, 1))
    # print((timeit.timeit(wrapped, number=20))/20)
    # print((timeit.timeit(wrapped2, number=20))/20)


if __name__ == '__main__':
    main()
