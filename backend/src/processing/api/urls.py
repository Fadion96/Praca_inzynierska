from django.urls import path

from .views import ProcessingFunctionRetrieveView, ProcessingFunctionView

urlpatterns = [
    path('', ProcessingFunctionView.as_view()),
    path('<pk>', ProcessingFunctionRetrieveView.as_view())
]