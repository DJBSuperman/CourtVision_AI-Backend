# predictor_api/serializers.py
from rest_framework import serializers
from .models import Team, TeamStats, Game, Prediction

class TeamSerializer(serializers.ModelSerializer):
    """Serializer for NBA teams"""
    class Meta:
        model = Team
        fields = ['code', 'name', 'created_at', 'updated_at']


class TeamStatsSerializer(serializers.ModelSerializer):
    """Serializer for team statistics"""
    team_name = serializers.CharField(source='team.name', read_only=True)
    
    class Meta:
        model = TeamStats
        fields = [
            'id', 'team', 'team_name', 'season', 'mp', 'fg', 'fga', 'fg_pct',
            'fg3', 'fg3a', 'fg3_pct', 'fg2', 'fg2a', 'fg2_pct', 'ft', 'fta',
            'ft_pct', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
            'pts', 'created_at', 'updated_at'
        ]


class GameSerializer(serializers.ModelSerializer):
    """Serializer for NBA games"""
    home_team_name = serializers.CharField(source='home_team.name', read_only=True)
    away_team_name = serializers.CharField(source='away_team.name', read_only=True)
    home_team_code = serializers.CharField(source='home_team.code', read_only=True)
    away_team_code = serializers.CharField(source='away_team.code', read_only=True)
    
    class Meta:
        model = Game
        fields = [
            'game_id', 'date', 'home_team', 'home_team_name', 'home_team_code',
            'away_team', 'away_team_name', 'away_team_code', 'home_score',
            'away_score', 'attendance', 'arena', 'status', 'created_at', 'updated_at'
        ]


class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for game predictions"""
    game_details = GameSerializer(source='game', read_only=True)
    
    class Meta:
        model = Prediction
        fields = [
            'id', 'game', 'game_details', 'home_win_probability',
            'prediction_time', 'home_team_advantage', 'away_team_advantage',
            'relevant_factors', 'actual_result', 'was_correct',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['prediction_time']


# predictor_api/serializers.py
class GamePredictionRequestSerializer(serializers.Serializer):
    home_team = serializers.CharField(max_length=3)
    away_team = serializers.CharField(max_length=3)
    
    def validate(self, data):
        if data['home_team'] == data['away_team']:
            raise serializers.ValidationError("Home and away teams must be different")
        return data