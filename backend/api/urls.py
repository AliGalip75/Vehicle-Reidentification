# api/urls.py
from django.urls import path
from .views import SearchView, MetricsView

urlpatterns = [
    path("search/", SearchView.as_view(), name="search"),
    path("metrics/", MetricsView.as_view(), name="metrics"),
]
