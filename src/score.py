import json
import os
import joblib
import pandas as pd
import numpy as np

model = None

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data)
        
        # Build features (same as training)
        exclude_cols = ['asin', 'reviewerID', 'overall', 'summary', 'reviewText', 
                        'reviewTime', 'title', 'brand', 'price', 'helpful', 'label',
                        'review_year', 'normalized_text', 'reviewText_clean']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0).values
        
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}