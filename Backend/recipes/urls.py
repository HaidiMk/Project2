from django.urls import path

from .views import RecommendationsView

app_name = "recipes"

urlpatterns = [
    path("recommendations/", RecommendationsView.as_view(), name="recommendations"),
]