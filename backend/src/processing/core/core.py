import inspect
import multiprocessing.pool
import numpy as np
from processing.core import algorithms as alg
from processing.core.utils import from_base64


def is_mod_function(mod, func):
    return inspect.isfunction(func) and inspect.getmodule(func) == mod


def dict_functions(mod):
    funct = {}
    for func in mod.__dict__.values():
        if is_mod_function(mod, func):
            funct.setdefault(func.__name__, func)
    return funct


def load_sources(path):
    inputs = path.get("inputs")
    for inp in inputs:
        source = inputs.get(inp).get("source")
        to = inputs.get(inp).get("to")
        if source is not None:
            img = from_base64(source)
            path.get("nodes")[to] = img
        else:
            raise TypeError(to + " - Nie załadowano obrazu po wczytaniu algorytmu")


def get_operation(img_key, path):
    operations = path.get("operations")
    for operation in operations:
        if operations.get(operation).get("to") == img_key:
            return operations.get(operation)
    return None


def do_operation(img_key, path, functions):
    if path.get("nodes").get(img_key) is None:
        operation = get_operation(img_key, path)
        func = operation.get("operation_name")
        print(func)
        images = []
        tmp = []
        for image in operation.get("from"):
            images.append(path.get("nodes").get(image))
            tmp.append((image, path, functions))
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
            for param in range(len(params)):
                if type(params[param]) is list:
                    params[param] = np.array(params[param], dtype="int")
                    print(params[param])
            res = functions[func](*images, *params)
        else:
            res = functions[func](*images)
        path.get("nodes")[img_key] = res


def check_adjacency(path):
    images = set(path.get("nodes").keys())
    ad_keys = set(path.get("adjacency").keys())
    ret_image = list(images - ad_keys)
    if len(ret_image) == 1:
        return True
    else:
        return False


def isCyclicUtil(v, visited, recStack, adjacency_list):
    visited[v] = True
    recStack[v] = True

    for neighbour in adjacency_list[v]:
        if not visited[neighbour]:
            if isCyclicUtil(neighbour, visited, recStack, adjacency_list):
                return True
        elif recStack[neighbour]:
            return True
    recStack[v] = False
    return False


def check_DAG(path):
    images = sorted(list(path.get("nodes").keys()))
    adjacency = path.get("adjacency")
    adjacency_list = [[] for i in range(len(images))]
    images_dict = {x: index for index, x in enumerate(images)}
    for key in adjacency:
        for value in adjacency[key]:
            adjacency_list[images_dict[key]].append(images_dict[value])
    visited = [False] * len(images)
    rec_stack = [False] * len(images)
    for node in range(len(images)):
        if not visited[node]:
            if isCyclicUtil(node, visited, rec_stack, adjacency_list):
                return True
    return False


def get_result_image_key(path):
    images = set(path.get("nodes").keys())
    ad_keys = set(path.get("adjacency").keys())
    ret_image = list(images - ad_keys)
    return ret_image[0]


def do_algorithm(path, user_functions):
    functions = dict_functions(alg)
    functions.update(user_functions)
    load_sources(path)
    result_image = get_result_image_key(path)
    do_operation(result_image, path, functions)
    return path.get("nodes").get(result_image)

