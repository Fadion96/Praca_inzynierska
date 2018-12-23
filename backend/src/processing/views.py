import base64

import cv2
from rest_framework.response import Response
from rest_framework.views import APIView

from processing.core.core import do_algorithm


class ProcessingView(APIView):

    def post(self, request, format=None):
        path = request.data.get("path")
        image = do_algorithm(path)
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        return Response({"ret": jpg_as_text})

