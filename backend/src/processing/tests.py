import base64
import json
import timeit
import numpy as np

from django.test import TestCase
from processing.core.processing import Processing
from processing.core.algorithms import Algorithms
from processing.core.utils import from_base64


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def get_json(path):
    with open(path) as json_file:
        json_str = json_file.read()
        json_data = json.loads(json_str)
    return json_data


def to_base64(filepath):
    with open(filepath, 'rb') as file:
        encoded_string = base64.b64encode(file.read())
        return encoded_string


class EngineTest(TestCase):
    def setUp(self):
        self.error_json_1 = get_json("processing/test_files/error_1.json")
        self.path_1 = Processing(self.error_json_1)
        self.error_json_2 = get_json("processing/test_files/error_2.json")
        self.path_2 = Processing(self.error_json_2)
        self.error_json_3 = get_json("processing/test_files/error_3.json")
        self.path_3 = Processing(self.error_json_3)
        self.good = get_json("processing/test_files/good.json")
        self.path_4 = Processing(self.good)
        self.bad_file = to_base64("processing/test_files/text.txt")

    def test_cycle(self):
        self.assertTrue(self.path_1.has_cycle())
        self.assertFalse(self.path_2.has_cycle())
        self.assertTrue(self.path_3.has_cycle())
        self.assertFalse(self.path_4.has_cycle())

    def test_result_count(self):
        self.assertTrue(self.path_1.has_one_result_image())
        self.assertFalse(self.path_2.has_one_result_image())
        self.assertTrue(self.path_3.has_one_result_image())
        self.assertTrue(self.path_4.has_one_result_image())

    def test_input(self):
        with self.assertRaises(TypeError):
            from_base64(self.bad_file)


class AlgorithmsTest(TestCase):
    def setUp(self):
        self.image_4k = from_base64(to_base64("processing/test_files/test_image_4k.jpg"))
        self.image_qhd = from_base64(to_base64("processing/test_files/test_image_qhd.jpg"))
        self.image_fhd = from_base64(to_base64("processing/test_files/test_image_fhd.jpg"))
        self.image_hd = from_base64(to_base64("processing/test_files/test_image_hd.jpg"))
        self.image_480 = from_base64(to_base64("processing/test_files/test_image_480.jpg"))

    def test_time(self):
        element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        sharpen = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]], dtype="int")
        source = np.array([1, 1])
        # print("Skala szarości: ")
        # wrapped = wrapper(Algorithms.grayscale_luma, self.image_fhd)
        # self.gray_fhd = wrapped()
        # fhd_time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        # wrapped = wrapper(Algorithms.grayscale_luma, self.image_4k)
        # self.gray_4k = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.grayscale_luma, self.image_qhd)
        # self.gray_qhd = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.grayscale_luma, self.image_hd)
        # self.gray_hd = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.grayscale_luma, self.image_480)
        # self.gray_480 = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # print("Progowanie Otsu: ")
        # wrapped = wrapper(Algorithms.otsu, self.gray_fhd)
        # self.bin_fhd = wrapped()
        # fhd_time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        # wrapped = wrapper(Algorithms.otsu, self.gray_4k)
        # self.bin_4k = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.otsu, self.gray_qhd)
        # self.bin_qhd = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.otsu, self.gray_hd)
        # self.bin_hd = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.otsu, self.gray_480)
        # self.bin_480 = wrapped()
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # print("Erozja: ")
        # wrapped = wrapper(Algorithms.erosion, self.bin_fhd, element, source)
        # fhd_time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        #
        # wrapped = wrapper(Algorithms.erosion, self.bin_4k, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # wrapped = wrapper(Algorithms.erosion, self.bin_qhd, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # wrapped = wrapper(Algorithms.erosion, self.bin_hd, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # wrapped = wrapper(Algorithms.erosion, self.bin_480, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # print("Dylacja: ")
        # wrapped = wrapper(Algorithms.dilation, self.bin_fhd, element, source)
        # fhd_time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        #
        # wrapped = wrapper(Algorithms.dilation, self.bin_4k, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # wrapped = wrapper(Algorithms.dilation, self.bin_qhd, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # wrapped = wrapper(Algorithms.dilation, self.bin_hd, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))

        # wrapped = wrapper(Algorithms.dilation, self.bin_480, element, source)
        # time = (timeit.timeit(wrapped, number=5) / 5) * 1000
        # print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # print("Negacja: ")
        # wrapped = wrapper(Algorithms.invert, self.image_fhd)
        # fhd_time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        # wrapped = wrapper(Algorithms.invert, self.image_4k)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.invert, self.image_qhd)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.invert, self.image_hd)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.invert, self.image_480)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        #
        # print("Wyrównywanie Histogramu: ")
        # wrapped = wrapper(Algorithms.histogram_equalization, self.image_fhd)
        # fhd_time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        # wrapped = wrapper(Algorithms.histogram_equalization, self.image_4k)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.histogram_equalization, self.image_qhd)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.histogram_equalization, self.image_hd)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        # wrapped = wrapper(Algorithms.histogram_equalization, self.image_480)
        # time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        # print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))

        print("Splot: ")
        wrapped = wrapper(Algorithms.convolution, self.image_fhd, sharpen)
        fhd_time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        print("FHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(fhd_time, 2), round(fhd_time / fhd_time * 100)))
        wrapped = wrapper(Algorithms.convolution, self.image_4k, sharpen)
        time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        print("4K & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        wrapped = wrapper(Algorithms.convolution, self.image_qhd, sharpen)
        time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        print("QHD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        wrapped = wrapper(Algorithms.convolution, self.image_hd, sharpen)
        time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        print("HD & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
        wrapped = wrapper(Algorithms.convolution, self.image_480, sharpen)
        time = (timeit.timeit(wrapped, number=10) / 10) * 1000
        print("480p & $ {:} $ ms & ${:}\%$ \\\\ \hline".format(round(time, 2), round(time / fhd_time * 100, 2)))
