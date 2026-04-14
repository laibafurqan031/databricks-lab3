import json
import os
import joblib
import pandas as pd
import numpy as np

model = None

def init():
    global model
    # The model is in a subfolder called "model_output"
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    model_path = os.path.join(model_dir, "model_output", "model.pkl")
    
    print(f"Looking for model at: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        
        # Handle different input formats
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        
        df = pd.DataFrame(data)
        
        # Feature extraction - use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_cols = ['asin', 'reviewerID', 'overall', 'label']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0).values
        
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
