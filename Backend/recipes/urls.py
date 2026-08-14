from django.urls import path

from .views import RecommendationsView, SearchView

app_name = "recipes"

urlpatterns = [
    path("recommendations/", RecommendationsView.as_view(), name="recommendations"),
    path("search/", SearchView.as_view(), name="search"),
]