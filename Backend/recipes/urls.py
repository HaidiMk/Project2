from django.urls import path

from .views import RecommendationsView, SearchView ,AlternativesView

app_name = "recipes"

urlpatterns = [
    path("recommendations/", RecommendationsView.as_view(), name="recommendations"),
    path("search/", SearchView.as_view(), name="search"),
    path("<int:recipe_id>/alternatives/", AlternativesView.as_view(), name="recipe-alternatives"),
]