# predictor_api/management/commands/train_from_csv.py
import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from predictor_api.ml_models.predictor import NBAPredictor

class Command(BaseCommand):
    help = 'Train the prediction model using a CSV file'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--csv',
            type=str,
            default='static/data/nba_team_data.csv',
            help='Path to the CSV training data file'
        )
    
    def handle(self, *args, **options):
        try:
            csv_path = options['csv']
            
            # Check if file path is absolute, if not use relative to project
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(settings.BASE_DIR, csv_path)
            
            # Verify the file exists
            if not os.path.exists(csv_path):
                self.stdout.write(self.style.ERROR(f"CSV file not found at {csv_path}"))
                return
                
            self.stdout.write(f"Loading training data from {csv_path}...")
            
            # Load the CSV data
            df = pd.read_csv(csv_path)
            
            # Check for required columns based on your CSV structure
            required_columns = ['home_win']
            diff_columns = [col for col in df.columns if '_diff' in col]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                self.stdout.write(self.style.ERROR(f"Missing required columns: {missing_cols}"))
                return
                
            if not diff_columns:
                self.stdout.write(self.style.ERROR("No differential columns found in data"))
                self.stdout.write(f"Available columns: {df.columns.tolist()}")
                return
                
            # Initialize the predictor
            predictor = NBAPredictor()
            
            # Train the model
            training_result = predictor.train_model(df)
            
            if training_result:
                self.stdout.write(self.style.SUCCESS("Model trained successfully"))
                self.stdout.write(f"Model accuracy: {training_result['accuracy']:.4f}")
                
                # Print top features
                self.stdout.write("Top predictive features:")
                for feature, importance in sorted(
                    training_result['feature_importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]:
                    self.stdout.write(f"- {feature}: {importance:.4f}")
            else:
                self.stdout.write(self.style.ERROR("Failed to train model"))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error training model: {str(e)}"))