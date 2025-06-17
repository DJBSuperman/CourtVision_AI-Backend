# predictor_api/management/commands/test_model.py
from django.core.management.base import BaseCommand
from predictor_api.ml_models.predictor import NBAPredictor

class Command(BaseCommand):
    help = 'Test the trained model with various scenarios'
    
    def handle(self, *args, **options):
        try:
            # Initialize the predictor
            predictor = NBAPredictor()
            
            # Load the model
            loaded = predictor.load_model()
            self.stdout.write(f"Model loaded: {loaded}")
            
            if not loaded:
                self.stdout.write(self.style.ERROR("Model not found. Train model first."))
                return
            
            # Start testing scenarios
            self.stdout.write("\n=== TESTING PREDICTION VARIATIONS ===")
            
            # Scenario 1: Teams with very similar stats (should be close to 50-60%)
            self.stdout.write("\n--- SCENARIO 1: EVENLY MATCHED TEAMS ---")
            home_stats_even = {
                'fg_pct': 0.47,
                'fg3_pct': 0.36,
                'ft_pct': 0.76,
                'ast': 24,
                'trb': 40,
                'stl': 7,
                'blk': 4,
                'tov': 13,
                'pts': 105,
                'home_team_name': 'Even Team A'
            }
            
            away_stats_even = {
                'fg_pct': 0.47,
                'fg3_pct': 0.35,
                'ft_pct': 0.75,
                'ast': 23,
                'trb': 39,
                'stl': 7,
                'blk': 4,
                'tov': 14,
                'pts': 104,
                'away_team_name': 'Even Team B'
            }
            
            prediction_even = predictor.predict_game(home_stats_even, away_stats_even)
            self.stdout.write(f"Home win probability: {prediction_even['home_win_probability']:.4f}")
            
            # Analyze prediction
            if 'error' not in prediction_even:
                analysis = predictor.analyze_prediction(
                    prediction_even,
                    home_stats_even['home_team_name'],
                    away_stats_even['away_team_name']
                )
                self.stdout.write(f"Prediction analysis: {analysis}")
            
            # Scenario 2: Home team with strong advantage
            self.stdout.write("\n--- SCENARIO 2: HOME TEAM ADVANTAGE ---")
            home_stats_strong = {
                'fg_pct': 0.52,
                'fg3_pct': 0.40,
                'ft_pct': 0.85,
                'ast': 28,
                'trb': 45,
                'stl': 9,
                'blk': 6,
                'tov': 10,
                'pts': 118,
                'home_team_name': 'Strong Home Team'
            }
            
            away_stats_weak = {
                'fg_pct': 0.43,
                'fg3_pct': 0.32,
                'ft_pct': 0.71,
                'ast': 19,
                'trb': 35,
                'stl': 5,
                'blk': 2,
                'tov': 16,
                'pts': 95,
                'away_team_name': 'Weak Away Team'
            }
            
            prediction_home = predictor.predict_game(home_stats_strong, away_stats_weak)
            self.stdout.write(f"Home win probability: {prediction_home['home_win_probability']:.4f}")
            
            # Analyze prediction
            if 'error' not in prediction_home:
                analysis = predictor.analyze_prediction(
                    prediction_home,
                    home_stats_strong['home_team_name'],
                    away_stats_weak['away_team_name']
                )
                self.stdout.write(f"Prediction analysis: {analysis}")
            
            # Scenario 3: Away team with strong advantage
            self.stdout.write("\n--- SCENARIO 3: AWAY TEAM ADVANTAGE ---")
            home_stats_weak = {
                'fg_pct': 0.42,
                'fg3_pct': 0.31,
                'ft_pct': 0.70,
                'ast': 18,
                'trb': 36,
                'stl': 5,
                'blk': 3,
                'tov': 17,
                'pts': 92,
                'home_team_name': 'Weak Home Team'
            }
            
            away_stats_strong = {
                'fg_pct': 0.51,
                'fg3_pct': 0.38,
                'ft_pct': 0.82,
                'ast': 26,
                'trb': 44,
                'stl': 8,
                'blk': 5,
                'tov': 11,
                'pts': 112,
                'away_team_name': 'Strong Away Team'
            }
            
            prediction_away = predictor.predict_game(home_stats_weak, away_stats_strong)
            self.stdout.write(f"Home win probability: {prediction_away['home_win_probability']:.4f}")
            
            # Analyze prediction
            if 'error' not in prediction_away:
                analysis = predictor.analyze_prediction(
                    prediction_away,
                    home_stats_weak['home_team_name'],
                    away_stats_strong['away_team_name']
                )
                self.stdout.write(f"Prediction analysis: {analysis}")
            
            # Summarize results
            self.stdout.write("\n=== SUMMARY OF PREDICTIONS ===")
            self.stdout.write(f"Evenly matched teams: {prediction_even['home_win_probability']:.4f}")
            self.stdout.write(f"Home team advantage: {prediction_home['home_win_probability']:.4f}")
            self.stdout.write(f"Away team advantage: {prediction_away['home_win_probability']:.4f}")
            
            # More reasonable expectations for basketball predictions
            even_range = (0.50, 0.65)  # Even matchup should be slightly favoring home (home court advantage)
            home_adv_range = (0.65, 0.85)  # Home advantage should be significant but not certain
            away_adv_range = (0.20, 0.40)  # Away advantage should still give home team some chance
            
            # Verify if predictions are within expected ranges
            even_ok = even_range[0] <= prediction_even['home_win_probability'] <= even_range[1]
            home_ok = home_adv_range[0] <= prediction_home['home_win_probability'] <= home_adv_range[1]
            away_ok = away_adv_range[0] <= prediction_away['home_win_probability'] <= away_adv_range[1]
            
            if even_ok and home_ok and away_ok:
                self.stdout.write(self.style.SUCCESS("\nPREDICTION MODEL IS WORKING CORRECTLY!"))
                self.stdout.write("The model is producing probabilities within expected ranges:")
                self.stdout.write(f"✓ Even matchup: {prediction_even['home_win_probability']:.4f} (expected {even_range[0]:.2f}-{even_range[1]:.2f})")
                self.stdout.write(f"✓ Home advantage: {prediction_home['home_win_probability']:.4f} (expected {home_adv_range[0]:.2f}-{home_adv_range[1]:.2f})")
                self.stdout.write(f"✓ Away advantage: {prediction_away['home_win_probability']:.4f} (expected {away_adv_range[0]:.2f}-{away_adv_range[1]:.2f})")
            else:
                self.stdout.write(self.style.WARNING("\nPREDICTION MODEL RESULTS:"))
                self.stdout.write(f"{'✓' if even_ok else '✗'} Even matchup: {prediction_even['home_win_probability']:.4f} (expected {even_range[0]:.2f}-{even_range[1]:.2f})")
                self.stdout.write(f"{'✓' if home_ok else '✗'} Home advantage: {prediction_home['home_win_probability']:.4f} (expected {home_adv_range[0]:.2f}-{home_adv_range[1]:.2f})")
                self.stdout.write(f"{'✓' if away_ok else '✗'} Away advantage: {prediction_away['home_win_probability']:.4f} (expected {away_adv_range[0]:.2f}-{away_adv_range[1]:.2f})")
                
                if not even_ok or not home_ok or not away_ok:
                    self.stdout.write(self.style.ERROR("\nSome predictions are outside of expected ranges."))
                    self.stdout.write("However, the model is now producing different probabilities based on team statistics,")
                    self.stdout.write("which is a significant improvement over the fixed 97/3 probabilities.")
            
            # Note about calibration
            self.stdout.write("\nNOTE: The model's raw probabilities are being calibrated to provide more realistic values.")
            self.stdout.write("Raw probabilities before calibration:")
            self.stdout.write(f"- Even matchup: {prediction_even.get('raw_probability', 'N/A')}")
            self.stdout.write(f"- Home advantage: {prediction_home.get('raw_probability', 'N/A')}")
            self.stdout.write(f"- Away advantage: {prediction_away.get('raw_probability', 'N/A')}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error testing model: {str(e)}"))