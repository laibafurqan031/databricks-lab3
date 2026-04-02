import argparse
import os
import time
import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def load_data(path):
    """Load parquet data"""
    return pd.read_parquet(path)

def create_labels(df):
    """Convert overall rating to binary label (1 if >=4, else 0)"""
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing!")
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df):
    """
    Combine all features from Lab 4.
    Your merged dataset already has all features.
    """
    # Identify feature columns
    # Exclude non-feature columns
    exclude_cols = ['asin', 'reviewerID', 'overall', 'summary', 'reviewText', 
                    'reviewTime', 'title', 'brand', 'price', 'helpful', 'label']
    
    # Get feature columns (all numeric columns not in exclude list)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns found!")
    
    print(f"Using {len(feature_cols)} features")
    
    # Convert to numpy array
    X = df[feature_cols].values
    
    # Handle any NaN values
    X = np.nan_to_num(X)
    
    return X

def evaluate(model, X, y, split_name):
    """Evaluate model and log metrics"""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    
    # Log metrics
    mlflow.log_metric(f"{split_name}_accuracy", acc)
    mlflow.log_metric(f"{split_name}_auc", auc)
    mlflow.log_metric(f"{split_name}_precision", prec)
    mlflow.log_metric(f"{split_name}_recall", rec)
    mlflow.log_metric(f"{split_name}_f1", f1)
    
    print(f"{split_name} - Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    
    return acc

def main():
    args = parse_args()
    start_time = time.time()
    
    # Start MLflow run
    mlflow.start_run()
    
    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    print("Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)
    
    print("Building features...")
    X_train = build_features(train_df)
    y_train = train_df["label"]
    X_val = build_features(val_df)
    y_val = val_df["label"]
    X_test = build_features(test_df)
    y_test = test_df["label"]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")
    
    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    
    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Total runtime: {runtime:.2f} seconds")
    
    mlflow.end_run()
    print("Done!")

if __name__ == "__main__":
    main()