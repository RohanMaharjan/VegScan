from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('scanner/', views.scanner_view, name='scanner'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('signup/', views.register_view, name='signup'),  # Alias for register
    path('logout/', views.logout_view, name='logout'),

    # API endpoint
    path('api/batch/', views.scan_batch_api, name='scan_batch_api'),
    path('api/slideshow/', views.slideshow_api, name='slideshow_api'),
    path('slideshow/image/<str:cls>/<str:fname>/', views.slideshow_image, name='slideshow_image'),
    path('api/feedback/', views.save_feedback_api, name='save_feedback_api'),
]
