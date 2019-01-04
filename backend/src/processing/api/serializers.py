from rest_framework import serializers

from processing.models import ProcessingFunction


class ProcessingFunctionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessingFunction
        fields = ('name', 'function', 'number_of_images', 'number_of_parameters', 'params', 'type')
