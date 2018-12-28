from django.urls import path

from processing.views import handle_processing

urlpatterns = [
    path('', handle_processing),
]
