# scraper/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.scraper_home, name='scraper-home'),
    path('run-scraper/', views.run_scraper, name='run-scraper'),
    path('data-status/', views.data_status, name='data-status'),
]