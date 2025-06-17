# model_diagnostic.py
# Run this as a management command to diagnose issues with the model

import os
import pandas as pd
import numpy as np
import joblib
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.pkl')

# Path to training data
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'data', 'nba_team_data.csv')

def inspect_model():
    """Inspect the existing model and its properties"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            logger.error("No model found at path: %s", MODEL_PATH)
            return False
            
        # Load model, scaler, and feature columns
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = pickle.load(open(FEATURE_COLS_PATH, 'rb'))
        
        # Display model info
        logger.info("Model type: %s", type(model).__name__)
        logger.info("Model parameters: %s", model.get_params())
        logger.info("Classes: %s", model.classes_)
        logger.info("Feature count: %d", len(feature_cols))
        logger.info("Features: %s", feature_cols)
        
        # Check feature importances
        importances = model.feature_importances_
        logger.info("Feature importances: %s", dict(zip(feature_cols, importances)))
        
        # Check class weights/balancing
        logger.info("Class distribution in model: %s", model.class_weight_)
        
        return model, scaler, feature_cols
    except Exception as e:
        logger.error("Error inspecting model: %s", str(e))
        return None

def inspect_training_data():
    """Inspect the training data for issues"""
    try:
        if not os.path.exists(DATA_PATH):
            logger.error("Training data not found at path: %s", DATA_PATH)
            return None
            
        # Load data
        df = pd.read_csv(DATA_PATH)
        logger.info("Data shape: %s", df.shape)
        
        # Check target distribution
        if 'home_win' in df.columns:
            home_win_count = df['home_win'].sum()
            logger.info("Target distribution: %d home wins out of %d games (%.2f%%)", 
                      home_win_count, len(df), 100 * home_win_count / len(df))
        else:
            logger.error("No 'home_win' column in training data")
            
        # Check for NaN values
        na_count = df.isna().sum().sum()
        logger.info("NaN values in dataset: %d", na_count)
        
        # Check feature distributions
        feature_cols = [col for col in df.columns if '_diff' in col]
        if feature_cols:
            feature_stats = df[feature_cols].describe().transpose()
            logger.info("Feature statistics:\n%s", feature_stats)
            
            # Check for features with very low variance
            low_var_features = feature_stats[feature_stats['std'] < 0.01]['std']
            if not low_var_features.empty:
                logger.warning("Features with low variance: %s", low_var_features.index.tolist())
                
            # Check for extreme values/outliers
            for col in feature_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col]
                if not outliers.empty:
                    logger.info("Outliers in %s: %d (%.2f%%)", 
                              col, len(outliers), 100 * len(outliers) / len(df))
        
        return df, feature_cols
    except Exception as e:
        logger.error("Error inspecting training data: %s", str(e))
        return None

def test_sample_predictions(model, scaler, feature_cols, df):
    """Test predictions on sample data points"""
    try:
        if model is None or df is None:
            return
            
        # Select some random samples from the dataset
        samples = df.sample(min(5, len(df)))
        logger.info("Testing %d random samples", len(samples))
        
        for i, sample in samples.iterrows():
            # Extract feature values
            X_sample = sample[feature_cols].values.reshape(1, -1)
            y_true = sample['home_win'] if 'home_win' in sample else None
            
            # Scale features
            X_scaled = scaler.transform(X_sample)
            
            # Make prediction
            y_pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            # Log results
            if y_true is not None:
                logger.info("Sample %d: True=%d, Pred=%d, Proba=%s, Features=%s", 
                          i, y_true, y_pred, proba, dict(zip(feature_cols, X_sample[0])))
            else:
                logger.info("Sample %d: Pred=%d, Proba=%s, Features=%s", 
                          i, y_pred, proba, dict(zip(feature_cols, X_sample[0])))
    except Exception as e:
        logger.error("Error testing sample predictions: %s", str(e))

def create_mock_prediction(team1_advantage=1, team2_advantage=-1):
    """
    Create mock prediction with controlled feature differences
    to see if model responds differently to different inputs
    """
    try:
        # Load model components
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = pickle.load(open(FEATURE_COLS_PATH, 'rb'))
        
        # Create a mock feature set where team1 is better in some stats, team2 in others
        mock_data = {}
        for i, col in enumerate(feature_cols):
            # Alternate advantages between teams
            if i % 2 == 0:
                mock_data[col] = team1_advantage  # Team 1 advantage
            else:
                mock_data[col] = team2_advantage  # Team 2 advantage
                
        # Create DataFrame
        mock_df = pd.DataFrame([mock_data])
        logger.info("Mock prediction data: %s", mock_df.to_dict(orient='records')[0])
        
        # Scale features
        X_scaled = scaler.transform(mock_df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        logger.info("Mock prediction result: %s with probabilities %s", prediction, probabilities)
        return prediction, probabilities
    except Exception as e:
        logger.error("Error creating mock prediction: %s", str(e))
        return None

def create_test_variations():
    """Create test cases with varying team advantages to see model response"""
    test_cases = [
        {"name": "Team 1 Strong Advantage", "team1": 2.0, "team2": -2.0},
        {"name": "Team 1 Slight Advantage", "team1": 0.5, "team2": -0.5},
        {"name": "Team 2 Strong Advantage", "team1": -2.0, "team2": 2.0},
        {"name": "Team 2 Slight Advantage", "team1": -0.5, "team2": 0.5},
        {"name": "Even Match", "team1": 0.1, "team2": -0.1},
    ]
    
    results = []
    for case in test_cases:
        logger.info("Testing case: %s", case["name"])
        pred, probs = create_mock_prediction(case["team1"], case["team2"])
        results.append({
            "case": case["name"],
            "prediction": pred,
            "home_win_prob": probs[1] if len(probs) > 1 else probs[0],
            "away_win_prob": probs[0] if len(probs) > 1 else 1 - probs[0]
        })
    
    for result in results:
        logger.info("Result for %s: Home win probability = %.4f", 
                  result["case"], result["home_win_prob"])
    
    return results

def run_diagnostics():
    """Run all diagnostic tests"""
    logger.info("=== Starting model diagnostics ===")
    
    # Step 1: Inspect existing model
    logger.info("--- Inspecting model ---")
    model_info = inspect_model()
    if not model_info:
        logger.error("Unable to inspect model. Make sure model files exist.")
        return
        
    model, scaler, feature_cols = model_info
    
    # Step 2: Inspect training data
    logger.info("--- Inspecting training data ---")
    data_info = inspect_training_data()
    if not data_info:
        logger.error("Unable to inspect training data. Make sure CSV file exists.")
        return
        
    df, _ = data_info
    
    # Step 3: Test sample predictions
    logger.info("--- Testing sample predictions ---")
    test_sample_predictions(model, scaler, feature_cols, df)
    
    # Step 4: Create mock predictions with varying advantages
    logger.info("--- Testing prediction variations ---")
    test_results = create_test_variations()
    
    logger.info("=== Diagnostics complete ===")
    
    # Return summary
    return {
        "model_exists": model is not None,
        "data_exists": df is not None,
        "test_results": test_results
    }

if __name__ == "__main__":
    run_diagnostics()