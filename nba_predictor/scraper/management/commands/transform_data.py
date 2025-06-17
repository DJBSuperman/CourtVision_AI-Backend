# predictor_api/management/commands/transform_data.py
import os
import pandas as pd
from django.core.management.base import BaseCommand
from predictor_api.models import Team, TeamStats, Game

class Command(BaseCommand):
    help = 'Transform scraped NBA data into training format for the ML model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default='nba_team_data.csv',
            help='Output file path for the transformed data'
        )
    
    def handle(self, *args, **options):
        self.stdout.write("Transforming NBA data for model training...")
        
        # Get output path
        output_file = options['output']
        
        # Get data from database
        games = Game.objects.filter(status='finished').order_by('date')
        
        if not games:
            self.stdout.write(self.style.ERROR("No finished games found in database"))
            return
            
        self.stdout.write(f"Processing {len(games)} finished games...")
        
        # Create dataframe for transformed data
        transformed_data = []
        
        for game in games:
            try:
                # Get teams
                home_team = game.home_team
                away_team = game.away_team
                
                # Get team stats
                try:
                    home_stats = TeamStats.objects.get(team=home_team, season="2024-2025")
                    away_stats = TeamStats.objects.get(team=away_team, season="2024-2025")
                except TeamStats.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f"Stats not found for game {game.game_id}"))
                    continue
                
                # Determine if home team won
                home_win = 1 if game.home_score > game.away_score else 0
                
                # Calculate differentials
                game_data = {
                    'game_id': game.game_id,
                    'date': game.date,
                    'home_team': home_team.code,
                    'away_team': away_team.code,
                    'home_score': game.home_score,
                    'away_score': game.away_score,
                    'home_win': home_win,
                    
                    # Create stat differentials (home - away)
                    'fg_pct_diff': home_stats.fg_pct - away_stats.fg_pct,
                    'fg3_pct_diff': home_stats.fg3_pct - away_stats.fg3_pct,
                    'ft_pct_diff': home_stats.ft_pct - away_stats.ft_pct,
                    'ast_diff': home_stats.ast - away_stats.ast,
                    'reb_diff': home_stats.trb - away_stats.trb,
                    'stl_diff': home_stats.stl - away_stats.stl,
                    'blk_diff': home_stats.blk - away_stats.blk,
                    'to_diff': home_stats.tov - away_stats.tov,
                    'pts_diff': home_stats.pts - away_stats.pts,
                }
                
                transformed_data.append(game_data)
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing game {game.game_id}: {str(e)}"))
        
        if not transformed_data:
            self.stdout.write(self.style.ERROR("No data was transformed"))
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(transformed_data)
        
        # Save to CSV
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            df.to_csv(output_file, index=False)
            self.stdout.write(self.style.SUCCESS(f"Transformed data saved to {output_file}"))
            
            # Also save to static directory for API access
            static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'static', 'data')
            if not os.path.exists(static_path):
                os.makedirs(static_path)
                
            static_file = os.path.join(static_path, 'nba_team_data.csv')
            df.to_csv(static_file, index=False)
            self.stdout.write(self.style.SUCCESS(f"Transformed data saved to static directory: {static_file}"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error saving transformed data: {str(e)}"))