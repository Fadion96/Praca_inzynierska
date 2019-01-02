import base64
import inspect
import os
import py_compile

import cv2
from rest_framework.response import Response
from rest_framework.decorators import api_view
from importlib.machinery import SourceFileLoader
from processing.core.core import do_algorithm, check_adjacency, check_DAG
from processing.models import ProcessingFunction, UserFunction, Session


@api_view(['POST'])
def handle_processing(request):
    path = request.data.get("path")
    if check_adjacency(path):
        if not check_DAG(path):
            try:
                user_functions_keys = request.data.get("user_functions")
                user_functions_dict = {}
                for key in user_functions_keys:
                    user_func_object = UserFunction.objects.get(pk=key)
                    function_name = user_func_object.name
                    module = SourceFileLoader(function_name, str(user_func_object.file)).load_module()
                    if hasattr(module, function_name):
                        func = getattr(module, function_name)
                        user_functions_dict.setdefault(func.__name__, func)
                image = do_algorithm(path, user_functions_dict)
                retval, buffer = cv2.imencode('.jpg', image)
                jpg_as_text = base64.b64encode(buffer)
                return Response({"ret": jpg_as_text})
            except Exception as e:
                return Response({'error': "Błąd: " + e.args[0]}, status=400)
        else:
            return Response({'error': "Cykl w algorytmie"}, status=400)
    else:
        return Response({'error': "Błąd: Wynikiem algorytmu może być tylko jeden obraz"}, status=400)


@api_view(['POST'])
def handle_file_upload(request):
    file = request.FILES.get("file")
    session_id = request.POST.get("session_id")
    name = file.name
    mod_name, file_ext = os.path.splitext(os.path.split(name)[-1])
    user_function = UserFunction.objects.create_user_function(session_id, mod_name, file)
    try:
        x = py_compile.compile(str(user_function.file), doraise=True)
    except py_compile.PyCompileError:
        return Response({'error': "Błąd: Błąd w pliku"},
                        status=400)
    module = SourceFileLoader(mod_name, str(user_function.file)).load_module()
    if hasattr(module, mod_name):
        func = getattr(module, mod_name)
        args = inspect.getfullargspec(func).args
        number_of_images = len([x for x in args if x.startswith('img')])
        number_of_params = len([x for x in args if not x.startswith('img')])
        return Response({
            "function_id": user_function.id,
            "function": {
                "name": func.__name__,
                "function": func.__name__,
                "number_of_images": number_of_images,
                "number_of_params": number_of_params,
                "params": None
            }
        })
    else:
        return Response({'error': "Błąd: Zła nazwa algorytmu, nazwa algorytmu powinna być taka sama jak nazwa pliku"},
                        status=400)


@api_view(['GET'])
def get_session_id(request):
    session = Session.objects.create_session()
    return Response({"id": session.session_id})


@api_view(['POST'])
def check_algorithm(request):
    path = request.data.get("path")
    if check_adjacency(path):
        if not check_DAG(path):
            return Response(path)
        else:
            return Response(
                {'error': "Błąd: Algorytm zawiera cykle"},
                status=400)
    else:
        return Response({'error': "Błąd: Wynikiem algorytmu może być tylko jeden obraz"}, status=400)