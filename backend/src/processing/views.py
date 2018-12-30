import base64

import cv2
from rest_framework.exceptions import ParseError
from rest_framework.response import Response
from rest_framework.decorators import api_view

from processing.core.core import do_algorithm, check_adjacency, check_DAG


@api_view(['POST'])
def handle_processing(request):
    path = request.data.get("path")
    if check_adjacency(path):
        if not check_DAG(path):
            try:
                image = do_algorithm(request.data.get("path"))
                retval, buffer = cv2.imencode('.jpg', image)
                jpg_as_text = base64.b64encode(buffer)
                return Response({"ret": jpg_as_text})
            except Exception as e:
                return Response({'error': "Błąd: " + e.args[0]}, status=400)
        else:
            return Response({'error': "Cykl w algorytmie"}, status=400)
    else:
        return Response({'error': "Błąd: Wynikiem algorytmu może być tylko jeden obraz"}, status=400)


