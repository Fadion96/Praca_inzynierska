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
        img = from_base64(source)
        path.get("nodes")[to] = img


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
            res = functions[func](*images, *params)
        else:
            res = functions[func](*images)
        path.get("nodes")[img_key] = res


def do_algorithm(path):
    functions = dict_functions(alg)
    load_sources(path)
    images = set(path.get("nodes").keys())
    ad_keys = set(path.get("adjacency").keys())
    ret_image = list(images - ad_keys)
    if len(ret_image) == 1:
        do_operation(ret_image[0], path, functions)
        return path.get("nodes").get(ret_image[0])
    else:
        return None



