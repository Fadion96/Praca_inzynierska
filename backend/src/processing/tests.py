import base64
import json

from django.test import TestCase
from processing.core.processing import Processing
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
