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
    # Hyperparameters for sweep
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations")
    return parser.parse_args()

def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing!")
    df["label"] = (df["overall"] >= 4).astype(int)
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def build_features(df):
    # Exclude non-feature columns
    exclude_cols = ['asin', 'reviewerID', 'overall', 'summary', 'reviewText', 
                    'reviewTime', 'title', 'brand', 'price', 'helpful', 'label',
                    'review_year', 'normalized_text', 'reviewText_clean']
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != 'label']
    
    # If no numeric features, try using all columns except excluded ones
    if len(feature_cols) == 0:
        feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'label']
    
    # Limit features to avoid memory issues (use top 500 features if too many)
    if len(feature_cols) > 500:
        print(f"Too many features ({len(feature_cols)}), limiting to top 500")
        # Get feature importance based on variance
        X_temp = df[feature_cols].fillna(0).values
        variances = np.var(X_temp, axis=0)
        top_indices = np.argsort(variances)[-500:]
        feature_cols = [feature_cols[i] for i in top_indices]
    
    print(f"Using {len(feature_cols)} features")
    
    X = df[feature_cols].fillna(0).values
    X = np.nan_to_num(X)
    
    return X

def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
    except:
        auc = 0.5
    
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
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
    
    mlflow.start_run()
    
    # Log hyperparameters
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)
    
    print("=" * 50)
    print("Starting training job...")
    print(f"Hyperparameters: C={args.C}, max_iter={args.max_iter}")
    print("=" * 50)
    
    print("\n1. Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    print("\n2. Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)
    
    print("\n3. Building features...")
    X_train = build_features(train_df)
    y_train = train_df["label"]
    X_val = build_features(val_df)
    y_val = val_df["label"]
    X_test = build_features(test_df)
    y_test = test_df["label"]
    
    print(f"\n4. Training set size: {X_train.shape}")
    print(f"   Validation set size: {X_val.shape}")
    print(f"   Test set size: {X_test.shape}")
    
    print("\n5. Training model...")
    model = LogisticRegression(max_iter=args.max_iter, random_state=42, C=args.C)
    model.fit(X_train, y_train)
    
    print("\n6. Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")
    
    print("\n7. Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    
    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"\n8. Total runtime: {runtime:.2f} seconds")
    
    mlflow.end_run()
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()