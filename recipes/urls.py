from django.urls import path
from . import views

urlpatterns = [
    path('classify/recipe/', views.recipe_view, name='recipe_view'),
]
