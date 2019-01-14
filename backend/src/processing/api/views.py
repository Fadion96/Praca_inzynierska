from rest_framework import viewsets
from processing.models import ProcessingFunction
from .serializers import ProcessingFunctionSerializer


class ProcessingFunctionViewSet(viewsets.ModelViewSet):
    """
    Zbiór widoków, obsługujący oglądanie, oraz edycje instancji zaimplementowanych algorytmów przetwarzania obrazów z bazy danych.
    """
    serializer_class = ProcessingFunctionSerializer
    queryset = ProcessingFunction.objects.all()
