import base64

import cv2
from rest_framework.exceptions import ParseError
from rest_framework.response import Response
from rest_framework.decorators import api_view

from processing.core.core import do_algorithm


@api_view(['POST'])
def handle_processing(request):
    image = do_algorithm(request.data.get("path"))
    if image is not None:
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        return Response({"ret": jpg_as_text})
    else:
        return Response({'error': "Błąd przetwarzania obrazka"}, status=400)
