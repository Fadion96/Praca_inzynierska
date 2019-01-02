from django.urls import path

from processing.views import handle_processing, handle_file_upload, get_session_id, check_algorithm

urlpatterns = [
    path('', handle_processing),
    path('upload', handle_file_upload),
    path('id', get_session_id),
    path('check', check_algorithm)
]
