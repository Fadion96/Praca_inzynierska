import multiprocessing
from inspect import getmembers, isroutine
import numpy as np
import cv2
from skimage.exposure import rescale_intensity


class Algorithms(object):

    @staticmethod
    def _get_dict_of_methods():
        return {name: func for name, func in getmembers(Algorithms, isroutine) if not name.startswith("_")}

    @staticmethod
    def grayscale_luma(img):
        """
        Funkcja, która przekształca obraz w modelu RGB (przechowywany jako BGR) na skalę szarości
        w standardzie ITU-R BT.709 dla HDTV.
        :param img: Tensor o wymiarach M x N x 3 przedstawiający liczbową reprezentacje obrazu w modelu RGB
        :return: Macierz o wymiarach M x N przedstawiajacy liczbową reprezentacje obrazu w skali szarości
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                w = np.array([[[0.07, 0.72, 0.21]]])
                gray = cv2.convertScaleAbs(np.sum(img * w, axis=2))
                return gray
            else:
                raise TypeError("Zmiana na skale szarości powinna być wykonywana tylko dla obrazów kolorowych")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
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
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                if isinstance(red_weight, (float, int)) and isinstance(green_weight, (float, int)) and isinstance(
                        blue_weight, (float, int)):
                    if 0.0 <= red_weight <= 1.0 and 0.0 <= green_weight <= 1.0 and 0.0 <= blue_weight <= 1.0:
                        if (red_weight + green_weight + blue_weight) == 1.0:
                            w = np.array([[[blue_weight, green_weight, red_weight]]])
                            gray = cv2.convertScaleAbs(np.sum(img * w, axis=2))
                            return gray
                        else:
                            raise ValueError("Suma wartości wag przy zamianie na skalę szarości musi wynosić 1")
                    else:
                        raise ValueError("Wartości wag przy zamianie na skalę szarości muszą być w zakresie [0,1]")
                else:
                    raise TypeError("Zły typ jednej z wag.")
            else:
                raise TypeError("Zmiana na skale szarości powinna być wykonywana tylko dla obrazów kolorowych")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def otsu(img):
        """
        Fukncja, która przekształca obraz w skali szarości na obraz binarny, za pomoca metody Otsu.
        :param img: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu w skali szarości.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                if not ((img == 0) | (img == 255)).all():
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
                else:
                    raise ValueError("Niepotrzeba binaryzacja obrazu binarnego")
            else:
                raise TypeError("Zmiana na obraz binarny powinna być wykonywana tylko dla obrazów w skali szarości")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def standard_threshold(img, threshold):
        """
        Fukncja, która przekształca obraz w skali szarości na obraz binarny z podaniem progu przez użytkownika.
        :param img: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu w skali szarości.
        :param threshold: Ustalony przez użytkownika próg.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                if not ((img == 0) | (img == 255)).all():
                    if isinstance(threshold, int):
                        if 0 <= threshold <= 255:
                            binary = img.copy()
                            binary[img > threshold] = 255
                            binary[img <= threshold] = 0
                            return binary
                        else:
                            raise ValueError("Wartość progu przy zamianie na obraz binarny musi być w zakresie [0,255]")
                    else:
                        raise TypeError("Próg powinien być liczbą naturalną w zakresie [0,255].")
                else:
                    raise ValueError("Niepotrzeba binaryzacja obrazu binarnego")
            else:
                raise TypeError("Zmiana na obraz binarny powinna być wykonywana tylko dla obrazów w skali szarości")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def dilation(img, structuring_element, anchor):
        """
        Fukcja, która na obrazie binarnym wykonuje morfologiczną operację dylacji.
        :param img: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest dylacja.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana dylacja.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej dylacji.
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                if ((img == 0) | (img == 255)).all():
                    if isinstance(structuring_element, np.ndarray):
                        if len(structuring_element.shape) == 2:
                            if ((structuring_element == 0) | (structuring_element == 1)).all():
                                if isinstance(anchor, np.ndarray):
                                    if len(anchor.shape) == 1:
                                        if anchor.size == 2:
                                            if 0 <= anchor[0] < structuring_element.shape[0] and 0 <= anchor[1] < \
                                                    structuring_element.shape[1]:
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
                                                            struct_window = structuring_element[
                                                                            struct_y_start:struct_y_end,
                                                                            struct_x_start:struct_x_end]
                                                            window = img[y_start:y_end, x_start:x_end] \
                                                                     & struct_window
                                                            result[i, j] = np.max(
                                                                window[np.where(struct_window == 1)]) * 255
                                                return result
                                            else:
                                                raise ValueError(
                                                    "Wartości elementów źródła powinny być w zakresie od 0 do rozmiaru elementu strukturalnego - 1")
                                        else:
                                            raise ValueError("Źródło powinno zawierać dwie liczby naturalne")
                                    else:
                                        raise TypeError(
                                            "Nieprawidłowy kształt żródła przekształcenia w elemencie strukturalnym")
                                else:
                                    raise TypeError("Zły typ żródła przekształcenia w elemencie strukturalnym")
                            else:
                                raise ValueError("Wartości w elemencie strukturalnym powinny wynosić 0 albo 1")
                        else:
                            raise TypeError("Element strukturalny powinien być macierzą")
                    else:
                        raise TypeError("Zły typ elementu strukturalnego")
                else:
                    raise ValueError("Obraz nie jest binarny")
            else:
                raise TypeError("Dylatacja powinna być wykonywana tylko dla obrazów binarnych")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def erosion(img, structuring_element, anchor):
        """
        Fukcja, która na obrazie binarnym wykonuje morfologiczną operację erozji.
        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                if ((img == 0) | (img == 255)).all():
                    if isinstance(structuring_element, np.ndarray):
                        if len(structuring_element.shape) == 2:
                            if ((structuring_element == 0) | (structuring_element == 1)).all():
                                if isinstance(anchor, np.ndarray):
                                    if len(anchor.shape) == 1:
                                        if anchor.size == 2:
                                            if 0 <= anchor[0] < structuring_element.shape[0] and 0 <= anchor[1] < \
                                                    structuring_element.shape[1]:
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
                                                            struct_window = structuring_element[
                                                                            struct_y_start:struct_y_end,
                                                                            struct_x_start:struct_x_end]
                                                            window = img[y_start:y_end, x_start:x_end] & struct_window
                                                            result[i, j] = np.min(
                                                                window[np.where(struct_window == 1)]) * 255
                                                return result
                                            else:
                                                raise ValueError(
                                                    "Wartości elementów źródła powinny być w zakresie od 0 do rozmiaru elementu strukturalnego - 1")
                                        else:
                                            raise ValueError("Źródło powinno zawierać dwie liczby naturalne")
                                    else:
                                        raise TypeError(
                                            "Nieprawidłowy kształt żródła przekształcenia w elemencie strukturalnym")
                                else:
                                    raise TypeError("Zły typ żródła przekształcenia w elemencie strukturalnym")
                            else:
                                raise ValueError("Wartości w elemencie strukturalnym powinny wynosić 0 albo 1")
                        else:
                            raise TypeError("Element strukturalny powinien być macierzą")
                    else:
                        raise TypeError("Zły typ elementu strukturalnego")
                else:
                    raise ValueError("Obraz nie jest binarny")
            else:
                raise TypeError("Erozja powinna być wykonywana tylko dla obrazów binarnych")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def opening(img, structuring_element, anchor):
        """
        Fukcja, która na obrazie binarnym wykonuje morfologiczną operację otwarcia, czyli wykonania operacji erozji, a nastepnie dylacji na obrazie.
        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywane jest otwarcie.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywane otwarcie.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanym otwarciu.
        """
        return Algorithms.dilation(Algorithms.erosion(img, structuring_element, anchor), structuring_element, anchor)

    @staticmethod
    def closing(img, structuring_element, anchor):
        """
        Fukcja, która na obrazie binarnym wykonuje morfologiczną operację otwarcia, czyli wykonania operacji dylacji, a nastepnie erozji na obrazie.
        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywane jest domkniecie.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywane domkniecie.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanym domknieciu.
        """
        return Algorithms.erosion(Algorithms.dilation(img, structuring_element, anchor), structuring_element, anchor)

    @staticmethod
    def hit_miss(img, structuring_element, anchor):
        """
        Fukcja, która na obrazie binarnym wykonuje morfologiczną operację Hit-or-miss, , którą to można przedstawić jako część wspólną erozji
        erozji obrazu z elementem strukturalnym oznaczającym "trafienia" oraz erozji odwróconego obrazu z elementem strukturalnym oznaczającym "pudła"
        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
        """
        if ((structuring_element == 0) | (structuring_element == 1) | (structuring_element == -1)).all():
            hit_element = structuring_element.copy()
            miss_element = structuring_element.copy()
            hit_element[structuring_element == 1] = 1
            hit_element[structuring_element != 1] = 0
            miss_element[structuring_element == -1] = 1
            miss_element[structuring_element != -1] = 0
            hit_img = Algorithms.erosion(img, hit_element, anchor)
            miss_img = Algorithms.erosion(Algorithms._invert_channel(img), miss_element, anchor)
            return hit_img & miss_img
        else:
            raise ValueError("Wartości w elemencie strukturalnym powinny wynosić -1, 0, 1")

    @staticmethod
    def morph_gradient(img, structuring_element, anchor):
        """

        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
        """
        dilation_dst = Algorithms.dilation(img, structuring_element, anchor)
        erosion_dst = Algorithms.erosion(img, structuring_element, anchor)
        return dilation_dst - erosion_dst

    @staticmethod
    def top_hat(img, structuring_element, anchor):
        """

        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
        """
        opening_dst = Algorithms.opening(img, structuring_element, anchor)
        return img - opening_dst

    @staticmethod
    def black_hat(img, structuring_element, anchor):
        """

        :param img:  Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego (wartosci {0, 255}).
        :param structuring_element: Element strukturalny - macierz najczęściej kwadratowa określająca zakres w jakim wykonywana jest erozja.
        :param anchor: Koordynaty punktu w elemencie strukturalnym, na którym to po nałożeniu elementu sktrukturalnego na obraz będzie wykonywana erozja.
        :return: Macierz o wymiarach M x N przedstawiajaca liczbową reprezentacje obrazu binarnego po wykonanej erozji.
        """
        closing_dst = Algorithms.closing(img, structuring_element, anchor)
        return closing_dst - img

    @staticmethod
    def invert(img):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                channels = []
                b = img[:, :, 0]
                g = img[:, :, 1]
                r = img[:, :, 2]
                channels.append(Algorithms._invert_channel(b))
                channels.append(Algorithms._invert_channel(g))
                channels.append(Algorithms._invert_channel(r))
                result = cv2.merge(channels)
            else:
                result = Algorithms._invert_channel(img)

            return result
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def change_brightness(img, value):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
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
            else:
                raise TypeError("Zmiana jasności powinna być wykonywana tylko dla obrazów w skali szarości")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def change_contrast(img, alpha):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                if isinstance(alpha, (float, int)):
                    height, width = img.shape
                    result = img.copy()
                    lut = Algorithms._make_lut_contrast(alpha)
                    for i in range(height):
                        for j in range(width):
                            result[i, j] = lut[img[i, j]]
                    return result
                else:
                    raise TypeError("Zły typ współczynnika.")
            else:
                raise TypeError("Zmiana jasności powinna być wykonywana tylko dla obrazów w skali szarości")
        else:
            raise TypeError("Zły typ obrazka")

    @staticmethod
    def stretching_histogram(img):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                height, width = img.shape
                result = img.copy()
                his, bins = np.histogram(img, np.arange(0, 257))
                non_zero = [index for index, item in enumerate(his) if item != 0]
                minimum = non_zero[0]
                maximum = non_zero[-1]
                lut = Algorithms._make_lut_stretching(minimum, maximum)
                for i in range(height):
                    for j in range(width):
                        result[i, j] = lut[img[i, j]]
                return result
            else:
                raise TypeError("Rozciąganie histogramu powinno być wykonywana tylko dla obrazów w skali szarości")
        else:
            raise TypeError("Zły typ obrazu")

    @staticmethod
    def multiplication(img, img_2):
        if isinstance(img, np.ndarray) and isinstance(img_2, np.ndarray):
            result = None
            if img.shape == img_2.shape:
                result = Algorithms._multiplication_channel(img, img_2)
            else:
                if img.shape[0] == img_2.shape[0] and img.shape[1] == img_2.shape[1]:
                    if len(img.shape) == 3:
                        channels = []
                        b = img[:, :, 0]
                        g = img[:, :, 1]
                        r = img[:, :, 2]
                        channels.append(Algorithms._multiplication_channel(b, img_2))
                        channels.append(Algorithms._multiplication_channel(g, img_2))
                        channels.append(Algorithms._multiplication_channel(r, img_2))
                        result = cv2.merge(channels)
                    elif len(img_2.shape) == 3:
                        channels = []
                        b = img_2[:, :, 0]
                        g = img_2[:, :, 1]
                        r = img_2[:, :, 2]
                        channels.append(Algorithms._multiplication_channel(img, b))
                        channels.append(Algorithms._multiplication_channel(img, g))
                        channels.append(Algorithms._multiplication_channel(img, r))
                        result = cv2.merge(channels)
                else:
                    tmp_img_2 = Algorithms.bilinear(img_2, img.shape[1], img.shape[0])
                    if img.shape == tmp_img_2.shape:
                        result = Algorithms._multiplication_channel(img, tmp_img_2)
                    elif len(img.shape) == 3:
                        channels = []
                        b = img[:, :, 0]
                        g = img[:, :, 1]
                        r = img[:, :, 2]
                        channels.append(Algorithms._multiplication_channel(b, tmp_img_2))
                        channels.append(Algorithms._multiplication_channel(g, tmp_img_2))
                        channels.append(Algorithms._multiplication_channel(r, tmp_img_2))
                        result = cv2.merge(channels)
                    elif len(tmp_img_2.shape) == 3:
                        channels = []
                        b = tmp_img_2[:, :, 0]
                        g = tmp_img_2[:, :, 1]
                        r = tmp_img_2[:, :, 2]
                        channels.append(Algorithms._multiplication_channel(img, b))
                        channels.append(Algorithms._multiplication_channel(img, g))
                        channels.append(Algorithms._multiplication_channel(img, r))
                        result = cv2.merge(channels)
            return result
        else:
            raise TypeError("Zły typ jednego z obrazów")

    @staticmethod
    def convolution(img, kernel):
        if isinstance(img, np.ndarray):
            if isinstance(kernel, np.ndarray):
                if len(kernel.shape) == 2:
                    if kernel.shape[0] == kernel.shape[1]:
                        if kernel.shape[0] % 2 == 1:
                            if len(img.shape) == 3:
                                b = img[:, :, 0]
                                g = img[:, :, 1]
                                r = img[:, :, 2]
                                pool = multiprocessing.Pool(processes=3)
                                channels = pool.starmap(Algorithms._convolution_channel,
                                                        [(b, kernel), (g, kernel), (r, kernel)])
                                pool.close()
                                result = cv2.merge(channels)
                            else:
                                result = Algorithms._convolution_channel(img, kernel)
                            return result
                        else:
                            raise TypeError("Jądro konwolucji powinno być krawdratem o rozmiarze nieparzystym")
                    else:
                        raise TypeError("Jądro konwolucji powinno być krawdratem o rozmiarze nieparzystym")
                else:
                    raise TypeError("Nieprawidłowy rozmiar jądra konwolucji")
            else:
                raise TypeError("Zły typ jądra konwolucji")
        else:
            raise TypeError("Zły typ obrazu")

    @staticmethod
    def histogram_equalization(img):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                b = img[:, :, 0]
                g = img[:, :, 1]
                r = img[:, :, 2]
                pool = multiprocessing.Pool(processes=3)
                channels = pool.map(Algorithms._histogram_equalization_channel, [b, g, r])
                pool.close()
                result = cv2.merge(channels)
            else:
                result = Algorithms._histogram_equalization_channel(img)

            return result
        else:
            raise TypeError("Zły typ obrazu")

    @staticmethod
    def bilinear(img, x, y):
        if isinstance(img, np.ndarray):
            if isinstance(x, int) and isinstance(y, int):
                if 0 < x and 0 < y:
                    if len(img.shape) == 3:
                        b = img[:, :, 0]
                        g = img[:, :, 1]
                        r = img[:, :, 2]
                        pool = multiprocessing.Pool(processes=3)
                        channels = pool.starmap(Algorithms._bilinear_channel, [(b, x, y), (g, x, y), (r, x, y), ])
                        pool.close()
                        result = cv2.merge(channels)
                    else:
                        result = Algorithms._bilinear_channel(img, x, y)
                    return result
                else:
                    raise ValueError("Wymiary obrazu po interpolacji powinny być większe od 0")
            else:
                raise TypeError("Zły typ wymiarów obrazu po interpolacji")
        else:
            raise TypeError("Zły typ obrazu")

    @staticmethod
    def _invert_channel(img):
        result = img.copy()
        result = 255 - result
        return result

    @staticmethod
    def _histogram_equalization_channel(img):
        result = img.copy()
        his, bins = np.histogram(img, np.arange(0, 257))
        cumulative_distribution = his.cumsum()
        lut = np.uint8(255 * cumulative_distribution / cumulative_distribution[-1])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                result[i, j] = lut[img[i, j]]
        return result

    @staticmethod
    def _make_lut_stretching(minimum, maximum):
        lut = [0] * 256
        for i in range(256):
            lut[i] = int((i - minimum) / (maximum - minimum) * 255)
        return lut

    @staticmethod
    def _make_lut_contrast(alpha):
        lut = np.array([0] * 256)
        for i in range(256):
            lut[i] = int(alpha * (i - 127) + 127)
        lut[lut > 255] = 255
        lut[lut < 0] = 0
        return lut

    @staticmethod
    def _convolution_channel(img, kernel):
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

    @staticmethod
    def _multiplication_channel(img, mask):
        result = (img / 255) * (mask / 255)
        return (result * 255).astype("uint8")

    @staticmethod
    def _bilinear_interp(img, x, y):
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

    @staticmethod
    def _bilinear_channel(img, x, y):
        height, width = img.shape[:2]
        result = np.zeros((y, x), dtype="uint8")
        height_ratio = height / y
        width_ratio = width / x

        for i in range(y):
            for j in range(x):
                result[i, j] = Algorithms._bilinear_interp(img, j * width_ratio, i * height_ratio)
        return result
