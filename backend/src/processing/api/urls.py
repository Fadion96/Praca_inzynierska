from .views import ProcessingFunctionViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'', ProcessingFunctionViewSet, basename='functions')
urlpatterns = router.urls