from django.urls import path

from processing.views import ProcessingView

urlpatterns = [
    path('', ProcessingView.as_view()),
]
