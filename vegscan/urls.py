from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # Include all scanner app URLs
    path('', include('scanner.urls')),
    # Recipes integration (isolated)
    path('', include('recipes.urls')),
]
