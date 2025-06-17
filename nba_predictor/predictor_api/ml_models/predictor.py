# predictor_api/ml_models/predictor.py
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import expit  # For sigmoid function
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBAPredictor:
    """
    NBA Game Prediction Model based on Random Forest
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        self.feature_cols_path = os.path.join(os.path.dirname(__file__), 'feature_cols.pkl')
    
    def load_model(self):
        """Load the trained model if available"""
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading existing model...")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.feature_cols = pickle.load(open(self.feature_cols_path, 'rb'))
                logger.info(f"Model loaded with {len(self.feature_cols)} features")
                return True
            else:
                logger.warning("No model found. Train model first.")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def train_model(self, df):
        """
        Train the Random Forest model on historical data
        """
        try:
            logger.info("Training new model...")
            
            # Select features and target - adjust to match your CSV format
            feature_cols = [col for col in df.columns if '_diff' in col]
            # Make sure 'home_win' is your target variable
            X = df[feature_cols]
            y = df['home_win']
            
            # Log info about features
            logger.info(f"Training on {len(feature_cols)} features: {feature_cols}")
            logger.info(f"Dataset size: {len(df)} games")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            rf_model = RandomForestClassifier(
                n_estimators=100,  # Reduced for faster training
                max_depth=10,      # Added to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model Accuracy: {accuracy:.4f}")
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, rf_model.feature_importances_.tolist()))
            
            # Save model, scaler, and feature columns
            joblib.dump(rf_model, self.model_path)
            joblib.dump(scaler, self.scaler_path)
            with open(self.feature_cols_path, 'wb') as f:
                pickle.dump(feature_cols, f)
            
            self.model = rf_model
            self.scaler = scaler
            self.feature_cols = feature_cols
            
            return {
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'feature_count': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def prepare_game_data(self, home_team_stats, away_team_stats):
        """
        Create differential features for prediction from team stats
        """
        try:
            # Convert string values to floats if needed
            for key in ['fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'trb', 'stl', 'blk', 'tov', 'pts']:
                if isinstance(home_team_stats.get(key), str):
                    home_team_stats[key] = float(home_team_stats[key])
                if isinstance(away_team_stats.get(key), str):
                    away_team_stats[key] = float(away_team_stats[key])
            
            # Create feature differences
            feature_mapping = {
                'fg_pct_diff': float(home_team_stats['fg_pct']) - float(away_team_stats['fg_pct']),
                'fg3_pct_diff': float(home_team_stats['fg3_pct']) - float(away_team_stats['fg3_pct']),
                'ft_pct_diff': float(home_team_stats['ft_pct']) - float(away_team_stats['ft_pct']),
                'ast_diff': float(home_team_stats['ast']) - float(away_team_stats['ast']),
                'reb_diff': float(home_team_stats['trb']) - float(away_team_stats['trb']),
                'stl_diff': float(home_team_stats['stl']) - float(away_team_stats['stl']),
                'blk_diff': float(home_team_stats['blk']) - float(away_team_stats['blk']),
                'to_diff': float(home_team_stats['tov']) - float(away_team_stats['tov']),
                'pts_diff': float(home_team_stats['pts']) - float(away_team_stats['pts'])
            }
            
            # Log the calculated feature differences
            logger.info(f"Feature differences: {feature_mapping}")
            
            # Create a DataFrame with just the differential features
            game_features = pd.DataFrame([feature_mapping])
            
            # Make sure all necessary features are present
            if self.feature_cols:
                for col in self.feature_cols:
                    if col not in game_features.columns:
                        logger.warning(f"Missing feature {col} in game data, setting to 0")
                        game_features[col] = 0
            
            # Log the final feature set being used
            logger.info(f"Final game features: {game_features.to_dict(orient='records')[0]}")
                    
            return game_features
        
        except Exception as e:
            logger.error(f"Error preparing game data: {str(e)}")
            return None
    
    def calibrate_probability(self, raw_prob, strength=0.5):
        """
        Calibrate overconfident probabilities to be more realistic
        
        Args:
            raw_prob: Raw probability from the model
            strength: Calibration strength (0.0-1.0)
                    0.0 = no calibration, 1.0 = maximum calibration (tends toward 0.5)
        
        Returns:
            Calibrated probability
        """
        # For very similar teams (small feature differences), use stronger calibration
        if 0.45 <= raw_prob <= 0.55:
            # Already close to 50%, no need for much calibration
            return raw_prob
        
        # For extreme predictions, apply non-linear calibration
        if raw_prob > 0.95 or raw_prob < 0.05:
            # Apply stronger calibration for extreme predictions
            adjusted_strength = strength * 1.2  # Increase calibration strength
        else:
            # Apply standard calibration for moderate predictions
            adjusted_strength = strength * 0.8  # Decrease calibration strength
        
        # Ensure strength is within bounds
        adjusted_strength = max(0.0, min(0.9, adjusted_strength))
        
        # Calculate distance from 0.5
        distance = abs(raw_prob - 0.5)
        
        # Apply calibration - reduce distance based on strength
        calibrated_distance = distance * (1 - adjusted_strength)
        
        # Reconstruct probability maintaining direction
        if raw_prob >= 0.5:
            return 0.5 + calibrated_distance
        else:
            return 0.5 - calibrated_distance

    
    def predict_game(self, home_team_stats, away_team_stats):
        """
        Predict the outcome of a game using the trained model
        """
        try:
            if not self.load_model():
                return {'error': 'Model not loaded'}
            
            # Use the prepared game data
            game_data = self.prepare_game_data(home_team_stats, away_team_stats)
            
            if game_data is None:
                return {'error': 'Failed to prepare game data'}
                
            # Scale features
            X_game_scaled = self.scaler.transform(game_data[self.feature_cols])
            
            # Log the scaled features for debugging
            logger.info(f"Scaled features for prediction: {X_game_scaled}")
            
            # Calculate dot product with feature importances for confidence assessment
            importance_weighted_sum = np.sum(np.abs(X_game_scaled[0] * self.model.feature_importances_))
            logger.info(f"Importance-weighted feature sum: {importance_weighted_sum}")
            
            # Make prediction - get probabilities for both classes
            probabilities = self.model.predict_proba(X_game_scaled)[0]
            logger.info(f"Raw prediction probabilities: {probabilities}")
            
            # Get correct probability for home win based on class labels
            # Check if 1 is in the classes and get its index
            if 1 in self.model.classes_:
                home_win_index = np.where(self.model.classes_ == 1)[0][0]
                win_prob = probabilities[home_win_index]
            else:
                # If classes are [0, 1] (common default)
                win_prob = probabilities[1]
            
            # Apply calibration to prevent extreme probabilities
            calibrated_prob = self.calibrate_probability(win_prob, strength=0.6)
            logger.info(f"Raw home win probability: {win_prob}")
            logger.info(f"Calibrated home win probability: {calibrated_prob}")
            
            # Get final prediction (0 = away win, 1 = home win)
            prediction = 1 if calibrated_prob > 0.5 else 0
            
            # Get team names
            home_team = home_team_stats.get('home_team_name', 'Home Team')
            away_team = away_team_stats.get('away_team_name', 'Away Team')
            
            # Calculate feature importances for this prediction
            feature_importances = self.model.feature_importances_
            
            # Get feature values for this game
            feature_values = game_data[self.feature_cols].values[0]
            
            # Calculate impact of each feature
            feature_impacts = []
            for i, col in enumerate(self.feature_cols):
                impact = feature_values[i] * feature_importances[i]
                team_favored = 'Home' if feature_values[i] > 0 else 'Away'
                feature_impacts.append({
                    'feature': col.replace('_diff', ''),
                    'impact': abs(impact),
                    'team_favored': team_favored if feature_values[i] != 0 else 'Neither'
                })
            
            # Sort by absolute impact
            top_factors = sorted(feature_impacts, key=lambda x: x['impact'], reverse=True)[:5]
            
            return {
                'home_win_probability': float(calibrated_prob),
                'raw_probability': float(win_prob),
                'prediction': int(prediction),
                'home_team': home_team,
                'away_team': away_team,
                'top_factors': top_factors
            }
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {'error': str(e)}
            
    def analyze_prediction(self, prediction_data, home_team_name, away_team_name):
        """
        Provide analysis and explanation of the prediction
        """
        win_prob = prediction_data['home_win_probability']
        
        # Confidence level description
        if win_prob > 0.8 or win_prob < 0.2:
            confidence = "very high"
        elif win_prob > 0.65 or win_prob < 0.35:
            confidence = "high"
        else:
            confidence = "moderate"
            
        # Determine predicted winner
        predicted_winner = home_team_name if win_prob > 0.5 else away_team_name
        win_probability = max(win_prob, 1-win_prob) * 100  # Convert to percentage
        
        # Build analysis
        analysis = {
            'predicted_winner': predicted_winner,
            'win_probability': win_probability,
            'confidence_level': confidence,
        }
        
        # Add context-specific analysis
        if 0.45 < win_prob < 0.55:
            analysis['summary'] = "This game appears to be very evenly matched."
        elif win_prob > 0.7:
            analysis['summary'] = f"{home_team_name} has a strong statistical advantage in this matchup."
        elif win_prob < 0.3:
            analysis['summary'] = f"{away_team_name} has a strong statistical advantage despite playing away."
        else:
            analysis['summary'] = f"The statistics slightly favor {predicted_winner}."
            
        # Add key factors from the prediction
        if 'top_factors' in prediction_data:
            home_advantages = []
            away_advantages = []
            
            for factor in prediction_data['top_factors']:
                if factor['team_favored'] == 'Home':
                    home_advantages.append(factor['feature'])
                elif factor['team_favored'] == 'Away':
                    away_advantages.append(factor['feature'])
            
            if home_advantages:
                analysis['home_team_advantages'] = home_advantages
            if away_advantages:
                analysis['away_team_advantages'] = away_advantages
                
        return analysis