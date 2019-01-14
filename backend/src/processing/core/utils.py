import base64

import numpy as np
import cv2


def load_image(path) -> np.ndarray:
    """
    Funkcja ładująca obraz do programu.
    :param path: ścieżka obrazu
    :return: Załadowany obraz.
    """
    img = cv2.imread(path)
    if isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Błąd, nie można załadować obrazka")


def to_image_string(image_filepath):
    """
    Funkcja ładująca plik w postaci kodowania base64
    :param image_filepath: ścieżka pliku
    :return: kodowanie base64
    """
    with open(image_filepath, 'rb') as image:
        encoded_string = base64.b64encode(image.read())
        return encoded_string


def  from_base64(base64_data):
    """
    Funkcja ładująca obraz w postaci kodu base64 do programu
    :param base64_data: kod base64 zawierający obraz
    :return: Załadowany obraz.
    """
    nparr = np.fromstring(base64.b64decode(base64_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Nie można załadować obrazka. Niewłaściwy typ")
