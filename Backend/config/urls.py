from django.contrib import admin
from django.urls import path ,include 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/accounts/', include('accounts.urls')),
    path('api/profiles/', include('profiles.urls')),
    path('api/recipes/', include('recipes.urls')),
]
