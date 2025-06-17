# In predictor_api/__init__.py
from predictor_api.ml_models.predictor import NBAPredictor
from .ml_models.predictor import NBAPredictor

# Create a global instance
global_predictor = NBAPredictor()
global_predictor.load_model()