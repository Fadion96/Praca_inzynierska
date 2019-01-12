import inspect
import multiprocessing.pool
import numpy as np
from processing.core.algorithms import Algorithms
from processing.core.utils import from_base64


class Processing(object):
    def __init__(self, path):
        self.nodes = path.get("nodes")
        self.operations = path.get("operations")
        print(self.operations)
        self.inputs = path.get("inputs")
        print(self.inputs)
        self.adjacency = path.get("adjacency")
        self.functions = self.set_functions()

    def load_sources(self):
        for input_image in self.inputs:
            source = self.inputs.get(input_image).get("source")
            to = self.inputs.get(input_image).get("to")
            if source is not None:
                img = from_base64(source)
                self.nodes[to] = img
            else:
                raise TypeError(to + " - Nie załadowano obrazu po wczytaniu algorytmu")

    def update_functions(self, user_functions):
        self.functions.update(user_functions)

    def get_result_image_key(self):
        images = set(self.nodes.keys())
        ad_keys = set(self.adjacency.keys())
        ret_image = list(images - ad_keys)
        return ret_image[0]

    def get_operation(self, img_key):
        for operation in self.operations:
            if img_key == self.operations.get(operation).get("to"):
                return self.operations.get(operation)

    def do_operation(self, img_key):
        if self.nodes.get(img_key) is None:
            operation = self.get_operation(img_key)
            func = operation.get("operation_name")
            images = []
            image_keys = []
            for image in operation.get("from"):
                images.append(self.nodes.get(image))
                image_keys.append(image)
            while any(x is None for x in images):
                pool = multiprocessing.pool.ThreadPool(processes=2)
                pool.map(self.do_operation, image_keys)
                pool.close()
                images = []
                image_keys = []
                for image in operation.get("from"):
                    images.append(self.nodes.get(image))
                    image_keys.append(image)
            params = operation.get("params")
            if params:
                for param in range(len(params)):
                    if type(params[param]) is list:
                        params[param] = np.array(params[param], dtype="int")
                        print(params[param])
                res = self.functions[func](*images, *params)
            else:
                res = self.functions[func](*images)
            self.nodes[img_key] = res

    def do_algorithm(self):
        self.load_sources()
        result_image = self.get_result_image_key()
        self.do_operation(result_image)
        return self.nodes.get(result_image)

    def has_one_result_image(self):
        images = set(self.nodes.keys())
        ad_keys = set(self.adjacency.keys())
        ret_image = list(images - ad_keys)
        if len(ret_image) == 1:
            return True
        else:
            return False

    def has_cycle(self):
        images = sorted(list(self.nodes.keys()))
        adjacency_list = [[] for _ in range(len(images))]
        images_dict = {x: index for index, x in enumerate(images)}
        for key in self.adjacency:
            for value in self.adjacency[key]:
                adjacency_list[images_dict[key]].append(images_dict[value])
        visited = [False] * len(images)
        stack = [False] * len(images)
        for node in range(len(images)):
            if not visited[node]:
                if self.__has_cycle(node, visited, stack, adjacency_list):
                    return True
        return False

    def __has_cycle(self, v, visited, stack, adjacency_list):
        visited[v] = True
        stack[v] = True

        for neighbour in adjacency_list[v]:
            if not visited[neighbour]:
                if self.__has_cycle(neighbour, visited, stack, adjacency_list):
                    return True
            elif stack[neighbour]:
                return True
        stack[v] = False
        return False

    @staticmethod
    def set_functions():
        return Algorithms._get_dict_of_methods()


