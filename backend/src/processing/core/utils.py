import base64

import numpy as np
import cv2
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("{:} ms".format((te - ts) * 1000))

        return result

    return timed


def load_image(path) -> np.ndarray:
    img = cv2.imread(path)
    if isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Błąd, nie można załadować obrazka")


def to_image_string(image_filepath):
    with open(image_filepath, 'rb') as image:
        encoded_string = base64.b64encode(image.read())
        return encoded_string


def from_base64(base64_data):
    nparr = np.fromstring(base64.b64decode(base64_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Nie można załadować obrazka. Niewłaściwy typ")
