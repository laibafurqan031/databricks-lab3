import requests
import json
import pandas as pd
import numpy as np
from azureml.core import Workspace
from sklearn.metrics import accuracy_score

ENDPOINT_URL = "https://amazon-review-endpoint-60301575.qatarcentral.inference.ml.azure.com/score"
API_KEY = "6eVkyrwovXO7qCa0Guib1N0W4jLVGrCqpvoNG4pPNRso2RryLPZlJQQJ99CDAAAAAAAAAAAAINFRAZML3RUg"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def main():
    print("Loading deployment dataset...")
    
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    
    import os
    os.makedirs("deploy_temp", exist_ok=True)
    datastore.download("deploy_temp", "Users/60301575/deploy_out/data.parquet", show_progress=True)
    
    df = pd.read_parquet("deploy_temp/Users/60301575/deploy_out/data.parquet")
    print(f"Loaded {len(df)} rows")
    
    exclude_cols = ['asin', 'reviewerID', 'overall', 'summary', 'reviewText', 
                    'reviewTime', 'title', 'brand', 'price', 'helpful', 'label',
                    'review_year', 'normalized_text', 'reviewText_clean']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(feature_cols) > 500:
        feature_cols = feature_cols[:500]
    
    X = df[feature_cols].fillna(0).values.tolist()
    y_true = (df['overall'] >= 4).astype(int).values
    
    print(f"Sending {len(X)} samples...")
    
    payload = {"data": X}
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        predictions = result.get("predictions", [])
        acc = accuracy_score(y_true, predictions)
        print(f"\nDeployment Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
