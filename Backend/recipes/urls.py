from django.urls import path

from .views import (
    AlternativesView,
    ExplanationView,
    RecipeDetailView,
    RecommendationsView,
    SearchView,
    MealPlannerView ,
    DashboardStatsView 
)

app_name = "recipes"

urlpatterns = [
    path("recommendations/", RecommendationsView.as_view(), name="recommendations"),
    path("search/", SearchView.as_view(), name="search"),
    path("<int:recipe_id>/alternatives/", AlternativesView.as_view(), name="recipe-alternatives"),
    path("<int:recipe_id>/explanation/", ExplanationView.as_view(), name="recipe-explanation"),
    path("<int:recipe_id>/", RecipeDetailView.as_view(), name="recipe-detail"),
    path("meal-planner/", MealPlannerView.as_view(), name="meal-planner"),  
    path("dashboard/stats/", DashboardStatsView.as_view(), name="dashboard-stats"),
]