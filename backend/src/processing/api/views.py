from rest_framework import viewsets
from processing.models import ProcessingFunction
from .serializers import ProcessingFunctionSerializer


class ProcessingFunctionViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing user instances.
    """
    serializer_class = ProcessingFunctionSerializer
    queryset = ProcessingFunction.objects.all()
