# predictor_api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router and register our viewsets
router = DefaultRouter()
router.register(r'teams', views.TeamViewSet)
router.register(r'team-stats', views.TeamStatsViewSet)
router.register(r'games', views.GameViewSet)
router.register(r'predictions', views.PredictionViewSet)

# The API URLs are determined automatically by the router
urlpatterns = [
    path('', include(router.urls)),
    path('train-model/', views.train_model, name='train-model'),
    path('predict-custom/', views.predict_custom_matchup, name='predict-custom'),
    path('upcoming-games/', views.get_upcoming_games, name='upcoming-games'),
]