from rest_framework.generics import ListAPIView, RetrieveAPIView
from processing.models import ProcessingFunction
from .serializers import ProcessingFunctionSerializer


class ProcessingFunctionView(ListAPIView):
    queryset = ProcessingFunction.objects.all()
    serializer_class = ProcessingFunctionSerializer


class ProcessingFunctionRetrieveView(RetrieveAPIView):
    queryset = ProcessingFunction.objects.all()
    serializer_class = ProcessingFunctionSerializer
