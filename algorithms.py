import numpy as np
import cv2


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


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
    img_copy = img.copy()
    img_copy = img_copy // 255
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
    miss_img = erosion(invert(img), miss_element, anchor)
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


def invert(img):
    result = img.copy()
    result = 255 - result
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


def histogram_equalization(img):
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
    lut = [0] * 256
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


# rescaling, mnozenie przez maske, kernele
# do gui rozwidlenie zlaczenie
# weryfikacja typow (klasa)
# semafor na czekanie sciezek
#

def main():
    img = cv2.imread('hd.jpg')
    gray = grayscale_luma(img)
    his = histogram_equalization(gray)
    # con = change_brightness(change_brightness(gray, 100), -50)
    # contr = stretching_histogram(con)
    cv2.imshow("gray", gray)
    cv2.imshow('hist', his)
    cv2.waitKey()
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
