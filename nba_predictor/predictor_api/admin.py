# predictor_api/admin.py
from django.contrib import admin
from django.contrib import messages
from django.shortcuts import redirect
from django.urls import path
from django.template.response import TemplateResponse
from .models import Team, TeamStats, Game, Prediction
from .ml_models.predictor import NBAPredictor
import pandas as pd
import os

class PredictionAdmin(admin.ModelAdmin):
    list_display = ('game', 'home_win_probability', 'prediction_time')
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('train-model/', self.admin_site.admin_view(self.train_model_view), name='train-model'),
        ]
        return custom_urls + urls
    
    def train_model_view(self, request):
        if request.method == 'POST':
            try:
                # Load CSV data
                csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nba_team_data.csv')
                df = pd.read_csv(csv_path)
                
                # Train model
                predictor = NBAPredictor()
                result = predictor.train_model(df)
                
                if result:
                    messages.success(request, f"Model trained successfully. Accuracy: {result['accuracy']:.2%}")
                else:
                    messages.error(request, "Failed to train model")
            except Exception as e:
                messages.error(request, f"Error training model: {str(e)}")
            
            return redirect('admin:predictor_api_prediction_changelist')
        
        context = {
            'title': 'Train Prediction Model',
            'opts': self.model._meta,
        }
        return TemplateResponse(request, 'admin/train_model.html', context)

admin.site.register(Prediction, PredictionAdmin)