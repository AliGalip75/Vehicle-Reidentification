# backend/urls.py

from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("api/", include("api.urls")),
]

urlpatterns += static("/", document_root=settings.MEDIA_ROOT)
