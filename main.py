import copy
from importlib.machinery import SourceFileLoader
import multiprocessing
import multiprocessing.pool
import os

import numpy as np
import json
import cv2
from utils import to_image_string, from_base64
import timeit
import algorithms as alg
import inspect

TEST_DAG = {
    "adjacency": {
        "img_1": ["img_2", "img_4"],
        "img_2": ["img_3"],
        "img_3": ["img_10"],
        "img_4": ["img_5", "img_7"],
        "img_5": ["img_6"],
        "img_6": ["img_9"],
        "img_7": ["img_8"],
        "img_8": ["img_9"],
        "img_9": ["img_10"],
        "img_10": ["img_11"],
        "img_11": ["img_12"],
    },
    "nodes": {
        "img_1": None,
        "img_2": None,
        "img_3": None,
        "img_4": None,
        "img_5": None,
        "img_6": None,
        "img_7": None,
        "img_8": None,
        "img_9": None,
        "img_10": None,
        "img_11": None,
        "img_12": None,
    },
    "operations": {
        "operation1": {
            "operation_name": "convolution",
            "from": ["img_1"],
            "to": "img_2",
            "params": [np.array((
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]), dtype="int")]
        },
        "operation2": {
            "operation_name": "histogram_equalization",
            "from": ["img_2"],
            "to": "img_3",
            "params": []
        },
        "operation3": {
            "operation_name": "grayscale_luma",
            "from": ["img_1"],
            "to": "img_4",
            "params": []
        },
        "operation4": {
            "operation_name": "convolution",
            "from": ["img_4"],
            "to": "img_5",
            "params": [np.array((
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]), dtype="int")]
        },
        "operation5": {
            "operation_name": "otsu",
            "from": ["img_5"],
            "to": "img_6",
            "params": []
        },
        "operation6": {
            "operation_name": "change_contrast",
            "from": ["img_4"],
            "to": "img_7",
            "params": [0.7]
        },
        "operation7": {
            "operation_name": "invert",
            "from": ["img_7"],
            "to": "img_8",
            "params": []
        },
        "operation8": {
            "operation_name": "multiplication",
            "from": ["img_6", "img_8"],
            "to": "img_9",
            "params": []
        },
        "operation9": {
            "operation_name": "multiplication",
            "from": ["img_3", "img_9"],
            "to": "img_10",
            "params": []
        },
        "operation10": {
            "operation_name": "invert",
            "from": ["img_10"],
            "to": "img_11",
            "params": []
        },
        "operation11": {
            "operation_name": "histogram_equalization",
            "from": ["img_11"],
            "to": "img_12",
            "params": []
        }
    },
    "inputs": {
        "input_1": {
            "source": "nosacz1.jpg",
            "to": "img_1"
        }
    }
}

DAG = copy.deepcopy(TEST_DAG)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def is_mod_function(mod, func):
    return inspect.isfunction(func) and inspect.getmodule(func) == mod


def dict_functions(mod):
    funct = {}
    for func in mod.__dict__.values():
        if is_mod_function(mod, func):
            funct.setdefault(func.__name__, func)
    return funct


def get_operation(img_key, path):
    operations = path.get("operations")
    for operation in operations:
        if operations.get(operation).get("to") == img_key:
            return operations.get(operation)
    return None


def do_operation(img_key, path):
    if path.get("nodes").get(img_key) is None:
        operation = get_operation(img_key, path)
        func = operation.get("operation_name")
        images = []
        tmp = []
        for image in operation.get("from"):
            images.append(path.get("nodes").get(image))
            tmp.append((image, path))
        while any(x is None for x in images):
            pool = multiprocessing.pool.ThreadPool(processes=2)
            pool.starmap(do_operation, tmp)
            pool.close()
            images = []
            tmp = []
            for image in operation.get("from"):
                images.append(path.get("nodes").get(image))
                tmp.append((image, path))
        params = operation.get("params")
        if params:
            if type(params[0]) is list:
                params[0] = np.array(params[0], dtype="int")
            res = functions[func](*images, *params)
        else:
            res = functions[func](*images)
        path.get("nodes")[img_key] = res


def load_sources(path):
    inputs = path.get("inputs")
    for inp in inputs:
        source = inputs.get(inp).get("source")
        to = inputs.get(inp).get("to")
        img = from_base64(source)
        path.get("nodes")[to] = img


def do_algorithm(path):
    load_sources(path)
    images = set(path.get("nodes").keys())
    ad_keys = set(path.get("adjacency").keys())
    ret_image = list(images - ad_keys)
    if len(ret_image) == 1:
        do_operation(ret_image[0], path)


def get_json(path):
    with open(path) as json_file:
        json_str = json_file.read()
        print(json_str)
        json_data = json.loads(json_str)
    return json_data


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


# do gui rozwidlenie zlaczenie
# weryfikacja typow (klasa)

def load_from_file(filepath):
    mod_name, file_ext = os.path.splitext(os.path.split(filepath)[-1])
    py_mod = SourceFileLoader(mod_name, filepath).load_module()
    if hasattr(py_mod, mod_name):
        func = getattr(py_mod, mod_name)
        user_func.setdefault(func.__name__, func)


functions = dict_functions(alg)
user_func = {}


def main():
    # własne skrypty (weź plik, wczytaj go, zapisz f do 2 słownika "nazwa": f, jezeli nie ma w moich, to z 2.
    # Zapis ścieżki bieremy tego jsona input na none (albo z inputem) tak samo algorytmy
    # zapis zdjecia
    # no i gui i testy

    path = get_json("path.json")
    #
    # do_algorithm(path)
    # # wrapped = wrapper(do_algorithm, TEST_DAG)
    # # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # for image in path.get("nodes"):
    #     if path.get("nodes").get(image) is not None:
    #         cv2.imshow(image, path.get("nodes").get(image))
    # cv2.waitKey()
    # print("Load image:")
    # img = from_base64(to_image_string("lena.png"))
    # img2 = from_base64(to_image_string("template2.png"))
    # load_from_file('adding.py')
    # res = user_func["adding"](img, img2)
    # cv2.imshow("test", res)
    # cv2.waitKey()
    # wrapped = wrapper(load_image, 'nosacz1.jpg')
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nGrayscale:")
    # gray = grayscale_luma(img)
    # wrapped = wrapper(grayscale_luma, img)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nGrayscale contrast:")
    # contgray = change_contrast(gray, 0.7)
    # wrapped = wrapper(change_contrast, gray, 0.7)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nNegate contrast:")
    # neg = invert(contgray)
    # wrapped = wrapper(invert, contgray)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # sharpen = np.array((
    #     [0, -1, 0],
    #     [-1, 5, -1],
    #     [0, -1, 0]), dtype="int")
    # laplacian = np.array((
    #     [0, 1, 0],
    #     [1, -4, 1],
    #     [0, 1, 0]), dtype="int")
    # print("\nGrayscale laplacian:")
    # lap = convolution(gray, laplacian)
    # wrapped = wrapper(convolution, gray, laplacian)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nLaplacian bin:")
    # lap_bin = otsu(lap)
    # wrapped = wrapper(otsu, lap)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nImg Sharp:")
    # sharp = convolution(img, sharpen)
    # wrapped = wrapper(convolution, img, sharpen)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nSharp HE:")
    # eq = histogram_equalization(sharp)
    # wrapped = wrapper(histogram_equalization, sharp)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nMulti1 Negate and lap_bin:")
    # multi = multiplication(neg, lap_bin)
    # wrapped = wrapper(multiplication, neg, lap_bin)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nMulti2 sharp HE and multi1:")
    # multi2 = multiplication(eq, multi)
    # wrapped = wrapper(multiplication, eq, multi)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nNegate multi2:")
    # inv = invert(multi2)
    # wrapped = wrapper(invert, multi2)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    # print("\nNegation HE:")
    # inveq = histogram_equalization(inv)
    # wrapped = wrapper(histogram_equalization, inv)
    # print("{:} ms".format(((timeit.timeit(wrapped, number=100)) / 100) * 1000))
    #
    # cv2.imwrite("prez2/start.jpg", img)
    # cv2.imwrite("prez2/gray.jpg", gray)
    # cv2.imwrite("prez2/conv.jpg", sharp)
    # cv2.imwrite("prez2/contgray.jpg", contgray)
    # cv2.imwrite("prez2/lap.jpg", lap)
    # cv2.imwrite("prez2/binlap.jpg", lap_bin)
    # cv2.imwrite("prez2/neg.jpg", neg)
    # cv2.imwrite("prez2/eq.jpg", eq)
    # cv2.imwrite("prez2/multi.jpg", multi)
    # cv2.imwrite("prez2/multi2.jpg", multi2)
    # cv2.imwrite("prez2/inv.jpg", inv)
    # cv2.imwrite("prez2/end.jpg", inveq)
    # cv2.imshow("gray", gray)
    # cv2.imshow("conv", sharp)
    # cv2.imshow("cont", contgray)
    # cv2.imshow("lap", lap)
    # cv2.imshow("binlap", lap_bin)
    # cv2.imshow("neg", neg)
    # cv2.imshow("eq", eq)
    # cv2.imshow("test", multi)
    # cv2.imshow("test2", multi2)
    # cv2.imshow("test3", inv)
    # cv2.imshow("end", inveq)
    # cv2.waitKey()
    # wrapped2 = wrapper(bin_erosion, binary, element, (1, 1))
    # print((timeit.timeit(wrapped2, number=20))/20)


if __name__ == '__main__':
    main()
