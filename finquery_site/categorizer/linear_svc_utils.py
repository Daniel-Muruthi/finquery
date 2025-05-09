import os
import joblib
import re

BASE = os.path.dirname(__file__)
PIPE_PATH = os.path.join(BASE, 'tuned_linear_svc_model.pkl')

# Load the full pipeline once
_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(PIPE_PATH)

def categorize_intent(text: str) -> str:
    """
    1. Clean text (lower + collapse whitespace)
    2. Let the pipeline handle vectorization + classification
    """
    # Ensure pipeline is loaded
    if _pipeline is None:
        _load_pipeline()
        
    # Clean text
    cleaned = re.sub(r'\s+', ' ', text.lower()).strip()
    
    # Pass a list of strings directly into the pipeline
    return _pipeline.predict([cleaned])[0]