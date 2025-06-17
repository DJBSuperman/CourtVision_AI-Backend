# predictor_api/models.py
from django.db import models
from django.utils import timezone

class Team(models.Model):
    """NBA Team information"""
    code = models.CharField(max_length=3, primary_key=True)  # Team code (e.g., 'LAL')
    name = models.CharField(max_length=50)  # Full team name
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.code})"
    
    class Meta:
        ordering = ['name']

class TeamStats(models.Model):
    """Stats for a specific team in a season"""
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='stats')
    season = models.CharField(max_length=9, default="2024-2025")  # Season in format YYYY-YYYY
    
    # Basic team stats - match your scraped data columns
    mp = models.FloatField(default=0)  # Minutes Played
    fg = models.FloatField(default=0)  # Field Goals
    fga = models.FloatField(default=0)  # Field Goal Attempts
    fg_pct = models.FloatField(default=0)  # Field Goal Percentage
    fg3 = models.FloatField(default=0)  # 3-Point Field Goals
    fg3a = models.FloatField(default=0)  # 3-Point Field Goal Attempts
    fg3_pct = models.FloatField(default=0)  # 3-Point Field Goal Percentage
    fg2 = models.FloatField(default=0)  # 2-Point Field Goals
    fg2a = models.FloatField(default=0)  # 2-Point Field Goal Attempts
    fg2_pct = models.FloatField(default=0)  # 2-Point Field Goal Percentage
    ft = models.FloatField(default=0)  # Free Throws
    fta = models.FloatField(default=0)  # Free Throw Attempts
    ft_pct = models.FloatField(default=0)  # Free Throw Percentage
    orb = models.FloatField(default=0)  # Offensive Rebounds
    drb = models.FloatField(default=0)  # Defensive Rebounds
    trb = models.FloatField(default=0)  # Total Rebounds
    ast = models.FloatField(default=0)  # Assists
    stl = models.FloatField(default=0)  # Steals
    blk = models.FloatField(default=0)  # Blocks
    tov = models.FloatField(default=0)  # Turnovers
    pf = models.FloatField(default=0)  # Personal Fouls
    pts = models.FloatField(default=0)  # Points
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.team.name} Stats ({self.season})"
    
    class Meta:
        unique_together = ('team', 'season')
        verbose_name_plural = 'Team Stats'

class Game(models.Model):
    """NBA Game information"""
    game_id = models.IntegerField(primary_key=True)
    date = models.DateField()
    home_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='home_games')
    away_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='away_games')
    home_score = models.IntegerField(null=True, blank=True)  # Null for future games
    away_score = models.IntegerField(null=True, blank=True)  # Null for future games
    attendance = models.IntegerField(null=True, blank=True)
    arena = models.CharField(max_length=100, blank=True)
    
    # Game status
    STATUS_CHOICES = (
        ('scheduled', 'Scheduled'),
        ('in_progress', 'In Progress'),
        ('finished', 'Finished'),
        ('cancelled', 'Cancelled'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.away_team.code} @ {self.home_team.code} ({self.date})"
    
    class Meta:
        ordering = ['-date']

class Prediction(models.Model):
    """Model predictions for games"""
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name='predictions')
    home_win_probability = models.FloatField()  # Probability of home team winning
    prediction_time = models.DateTimeField(default=timezone.now)
    
    # Prediction details
    home_team_advantage = models.TextField(blank=True)  # Key advantages of home team
    away_team_advantage = models.TextField(blank=True)  # Key advantages of away team
    relevant_factors = models.TextField(blank=True)  # Important factors in this prediction
    
    # After the game, mark if prediction was correct
    actual_result = models.BooleanField(null=True, blank=True)  # True if home team won
    was_correct = models.BooleanField(null=True, blank=True)  # True if prediction was correct
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        predicted_winner = self.game.home_team if self.home_win_probability > 0.5 else self.game.away_team
        return f"Prediction for {self.game}: {predicted_winner} ({self.home_win_probability:.2%})"
    
    class Meta:
        ordering = ['-prediction_time']