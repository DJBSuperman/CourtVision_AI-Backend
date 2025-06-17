# predictor_api/views.py
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated  # Also import this if you're using it
from django.shortcuts import get_object_or_404
from rest_framework.permissions import AllowAny
from predictor_api import global_predictor
import time
import pandas as pd
import os
import logging

from .models import Team, TeamStats, Game, Prediction
from .serializers import (
    TeamSerializer, TeamStatsSerializer, GameSerializer, 
    PredictionSerializer, GamePredictionRequestSerializer
)
from .ml_models.predictor import NBAPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamViewSet(viewsets.ModelViewSet):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer
    pagination_class = None  # Disable pagination for this viewset

class TeamStatsViewSet(viewsets.ModelViewSet):
    """API endpoint for team statistics"""
    queryset = TeamStats.objects.all()
    serializer_class = TeamStatsSerializer
    
    def get_queryset(self):
        """Allow filtering by team and season"""
        queryset = TeamStats.objects.all()
        team_code = self.request.query_params.get('team')
        season = self.request.query_params.get('season')
        
        if team_code:
            queryset = queryset.filter(team__code=team_code)
        if season:
            queryset = queryset.filter(season=season)
            
        return queryset

class GameViewSet(viewsets.ModelViewSet):
    """API endpoint for NBA games"""
    queryset = Game.objects.all()
    serializer_class = GameSerializer
    pagination_class = None
    
    def get_queryset(self):
        """Allow filtering by team, date range, and status"""
        queryset = Game.objects.all()
        team = self.request.query_params.get('team')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        status = self.request.query_params.get('status')
        
        if team:
            queryset = queryset.filter(home_team__code=team) | queryset.filter(away_team__code=team)
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)
        if status:
            queryset = queryset.filter(status=status)
            
        return queryset
    
    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None):
        """Generate prediction for a specific game"""
        game = self.get_object()
        
        # Get team statistics
        try:
            home_stats = TeamStats.objects.get(team=game.home_team, season="2024-2025")
            away_stats = TeamStats.objects.get(team=game.away_team, season="2024-2025")
        except TeamStats.DoesNotExist:
            return Response(
                {"error": "Team statistics not found for one or both teams"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Convert to dictionaries for the model
        home_stats_dict = TeamStatsSerializer(home_stats).data
        away_stats_dict = TeamStatsSerializer(away_stats).data
        
        # Initialize predictor and make prediction
        predictor = NBAPredictor()
        if not predictor.load_model():
            return Response(
                {"error": "Prediction model not available"}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        prediction_result = predictor.predict_game(home_stats_dict, away_stats_dict)
        if 'error' in prediction_result:
            return Response(
                {"error": prediction_result['error']}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Get analysis
        analysis = predictor.analyze_prediction(
            prediction_result, 
            game.home_team.name, 
            game.away_team.name
        )
        
        # Create or update prediction in database
        prediction, created = Prediction.objects.update_or_create(
            game=game,
            defaults={
                'home_win_probability': prediction_result['home_win_probability'],
                'home_team_advantage': ', '.join([f['feature'] for f in prediction_result['top_factors'] if f['team_favored'] == 'Home']),
                'away_team_advantage': ', '.join([f['feature'] for f in prediction_result['top_factors'] if f['team_favored'] == 'Away']),
                'relevant_factors': str(prediction_result['top_factors'])
            }
        )
        
        # Prepare response
        response_data = {
            'game': GameSerializer(game).data,
            'prediction': PredictionSerializer(prediction).data,
            'analysis': analysis
        }
        
        return Response(response_data)

class PredictionViewSet(viewsets.ModelViewSet):
    """API endpoint for game predictions"""
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    
    def get_queryset(self):
        """Allow filtering by game and date range"""
        queryset = Prediction.objects.all()
        game_id = self.request.query_params.get('game')
        
        if game_id:
            queryset = queryset.filter(game_id=game_id)
            
        return queryset



# predictor_api/views.py (update the train_model function)

@api_view(['POST'])
def train_model(request):
    """API endpoint to train the model with CSV data"""
    try:
        # Get the path to the CSV file
        csv_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'nba_team_data.csv')
        
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            return Response(
                {"error": "Training data file not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Create and train the model
        predictor = NBAPredictor()
        result = predictor.train_model(df)
        
        if result is None:
            return Response(
                {"error": "Failed to train model"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
        return Response({
            "message": "Model trained successfully",
            "details": result
        })
        
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_upcoming_games(request):
    """Get list of upcoming games with predictions if available"""
    games = Game.objects.filter(status='scheduled').order_by('date')[:10]
    
    results = []
    for game in games:
        game_data = GameSerializer(game).data
        
        # Get prediction if it exists
        try:
            prediction = Prediction.objects.get(game=game)
            prediction_data = PredictionSerializer(prediction).data
        except Prediction.DoesNotExist:
            prediction_data = None
            
        results.append({
            'game': game_data,
            'prediction': prediction_data
        })
        
    return Response(results)


@api_view(['POST'])
@permission_classes([AllowAny])
def predict_custom_matchup(request):
    """API endpoint to predict outcome of a custom matchup (not from database)"""
    # Add timing logs
    import time
    start_time = time.time()
    
    response_data = {}  # Initialize response_data at the beginning of the function
    
    serializer = GamePredictionRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    home_team_code = serializer.validated_data['home_team']
    away_team_code = serializer.validated_data['away_team']
    
    # Print for debugging
    print(f"Prediction request: {home_team_code} vs {away_team_code}")
    
    # Get team statistics
    try:
        print(f"Starting team lookup at {time.time() - start_time:.2f}s")
        home_team = get_object_or_404(Team, code=home_team_code)
        away_team = get_object_or_404(Team, code=away_team_code)
        home_stats = get_object_or_404(TeamStats, team=home_team, season="2024-2025")
        away_stats = get_object_or_404(TeamStats, team=away_team, season="2024-2025")
        print(f"Team lookup completed at {time.time() - start_time:.2f}s")
    except:
        return Response(
            {"error": "Team or statistics not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Convert to dictionaries for the model
    home_stats_dict = TeamStatsSerializer(home_stats).data
    away_stats_dict = TeamStatsSerializer(away_stats).data
    
    # Add team names for prediction display
    home_stats_dict['home_team_name'] = home_team.name
    away_stats_dict['away_team_name'] = away_team.name
    
    # Initialize predictor and make prediction
    print(f"Starting prediction at {time.time() - start_time:.2f}s")
    predictor = NBAPredictor()
    if not predictor.load_model():
        return Response(
            {"error": "Prediction model not available"}, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    prediction_result = predictor.predict_game(home_stats_dict, away_stats_dict)
    print(f"Prediction completed at {time.time() - start_time:.2f}s")
    
    if 'error' in prediction_result:
        return Response(
            {"error": prediction_result['error']}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    # Get analysis
    analysis = predictor.analyze_prediction(
        prediction_result, 
        home_team.name, 
        away_team.name
    )
    
    # Add home and away team advantages to the analysis
    if 'top_factors' in prediction_result:
        home_advantages = [factor['feature'] for factor in prediction_result['top_factors'] if factor['team_favored'] == 'Home']
        away_advantages = [factor['feature'] for factor in prediction_result['top_factors'] if factor['team_favored'] == 'Away']
        analysis['home_team_advantages'] = home_advantages
        analysis['away_team_advantages'] = away_advantages
    
    # Prepare response
    response_data = {
        'matchup': {
            'home_team': home_team.name,
            'away_team': away_team.name
        },
        'prediction': {
            'home_win_probability': prediction_result['home_win_probability'],
            'raw_probability': prediction_result.get('raw_probability', None),
            'predicted_winner': home_team.name if prediction_result.get('prediction', 0) == 1 else away_team.name,
            'top_factors': prediction_result.get('top_factors', [])  # Include top_factors array
        },
        'analysis': analysis
    }
    
    print(f"Total request time: {time.time() - start_time:.2f}s")
    return Response(response_data)