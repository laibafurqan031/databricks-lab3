# DSAI3202 – Assignment 2
## Model Training & Automation with Azure Machine Learning

---

## Assignment Objective

Build an end-to-end MLOps workflow on Azure ML including training, hyperparameter tuning,
model registration, online deployment, and endpoint testing using the Amazon Electronics
review dataset. A Logistic Regression model predicts positive vs negative reviews based on
pre-engineered features from Lab 4.

The full workflow:
code push → Azure DevOps pipeline → Azure ML training job → MLflow metrics → versioned model → deployed endpoint

---

## Repository Structure

```
├── src/
│ ├── train.py # Training script
│ ├── score.py # Scoring script for endpoint
│ └── invoke_endpoint.py # Endpoint invocation using deployment split
├── jobs/
│ ├── train_job.yml # Azure ML command job definition
│ ├── sweep_job.yml # Hyperparameter sweep job definition
│ └── deployment.yml # Online deployment configuration
├── env/
│ ├── conda.yml # Training environment
│ └── inference_conda.yml # Inference environment
├── azure-pipelines.yml # Azure DevOps CI pipeline
└── README.md
```

---

## Dataset

Amazon Electronics reviews with features engineered in Lab 4: SBERT embeddings,
TF-IDF vectors, sentiment scores, and review length statistics.

### Splits

| Split      | Proportion | Rows    | Notes                                      |
|------------|------------|---------|---------------------------------------------|
| Train      | 60%        | 225,937 | Model training                              |
| Validation | 15%        | 56,484  | Hyperparameter tuning and experiment comparison |
| Test       | 15%        | 56,485  | Final offline evaluation                    |
| Deployment | 10%        | 37,657  | Simulates production data (most recent reviews by review_year) |

**Label:** `1` if `overall >= 4` (positive review), else `0`.

---

## Features

Features were pre-computed in Lab 4 and consumed directly by the training script.
No feature engineering is performed at training time.

| Feature Type     | Description                                      |
|------------------|--------------------------------------------------|
| SBERT embeddings | Dense 384-dim vectors representing review text   |
| TF-IDF vectors   | Sparse word frequency representations            |
| Sentiment scores | Numeric polarity scores derived from review text |
| Length features  | Review character/word count statistics           |

---

## Model Choice

**Logistic Regression** was selected for the following reasons:

- Simple and fast to train, which is important when running experiments through CI pipelines
- Interpretable and easy to debug
- Works well with high-dimensional sparse feature inputs (TF-IDF)
- Competitive baseline for binary text classification tasks
- Low resource footprint, relevant for the efficiency bonus

---

## Hyperparameter Tuning (Sweep Job)

A sweep job was submitted via `jobs/sweep_job.yml` using random sampling over the following
search space:

| Hyperparameter | Type    | Range / Values |
|----------------|---------|---------------|
| C              | uniform | 0.001 – 10.0  |
| max_iter       | choice  | [100, 300, 500, 1000] |

- **Trials:** 8
- **Objective:** maximize `val_accuracy`
- **Best Run:** `frank_lettuce_3jd6g412z5_4`
- **Best Params:** `C ≈ 0.1`, `max_iter = 500`

The best hyperparameters were set as defaults in `train.py` and a final training run was
submitted before model registration.

---

## Final Model Performance

Trained using best hyperparameters from the sweep (C=0.1, max_iter=500), all features.

| Metric    | Train  | Validation | Test  |
|-----------|--------|------------|-------|
| Accuracy  | 85.6%  | 79.7%      | 79.7% |
| AUC       | 0.87   | 0.69       | 0.68  |
| F1 Score  | 91.4%  | 88.3%      | 88.3% |

**Training Runtime:** ~76 seconds (logged via MLflow as `training_runtime_seconds`)

**Note on AUC gap:** Train AUC (0.87) vs validation AUC (0.69) reflects moderate overfitting
on the dense SBERT dimensions. The model generalizes well enough on accuracy and F1, but
the AUC gap suggests the model is more confident on training data than it should be.
Regularization via C=0.1 partially addresses this.

---

## Azure DevOps CI Pipeline

The `azure-pipelines.yml` is configured to trigger on every push to the
`assignment2_model_training` branch.

**Pipeline steps:**
1. Check out repository
2. Install / update Azure ML CLI extension
3. Set Azure ML workspace defaults
4. Submit `jobs/train_job.yml` via `az ml job create`
5. Stream job logs until completion

Every push to the trigger branch automatically runs the full training workflow on Azure ML
compute without manual intervention.

---

## Model Registration

```bash
az ml model create \
  --name amazon-review-sentiment-model \
  --path azureml://jobs/<JOB_NAME>/outputs/model_output \
  --type custom_model
```

- Registered Model: `amazon-review-sentiment-model:1`
- Linked to the exact training job that produced it
- Hyperparameters, metrics, and outputs are fully traceable via MLflow

---

## Deployment

| Component   | Value |
|------------|------|
| Model       | amazon-review-sentiment-model:1 |
| Endpoint    | amazon-review-endpoint-60301575 |
| Deployment  | amazon-review-deployment |
| Instance    | Standard_F2s_v2 |
| Auth mode   | Key |
| Scoring URI | https://amazon-review-endpoint-60301575.qatarcentral.inference.ml.azure.com/score |

Deployed using `jobs/deployment.yml` with score.py as the scoring script.
The inference environment (env/inference_conda.yml) mirrors the training environment
without feature engineering dependencies.

---

## Endpoint Evaluation (Deployment Split)

The deployment split (10%, most recent reviews by review_year) was used to simulate
real production traffic via src/invoke_endpoint.py.

| Metric   | Test Set | Deployment Split (full) |
|----------|----------|-------------------------|
| Accuracy | 79.7%    | 79.1% |

The full deployment split (37,657 rows) yields 79.1% accuracy, closely matching
the test set performance. This confirms minimal data drift for this snapshot.

The deployment split performance was computed by running invoke_endpoint.py which sends
all deployment split rows to the endpoint and calculates accuracy against true labels.

---

## Bonus Question

> "There is one thing we are doing 'not correctly' in this assignment. What is it?"

Answer: The main issue is **data leakage during feature engineering**. The TF-IDF vectorizer and any scaling or transformation steps applied in Lab 4 were fit on the entire dataset before splitting. This means information from the validation and test sets influenced the feature representation used during training, which leads to overly optimistic evaluation results.

The correct approach is to fit all feature transformers (e.g., TF-IDF vectorizer, scalers) **only on the training set**, and then apply them to validation, test, and deployment sets without refitting.

A secondary issue is that although a deployment split based on `review_year` is used, the training, validation, and test splits are randomly shuffled rather than strictly time-based. In a real production system, a fully temporal split would better simulate future data and reduce potential leakage across time.

---

## Cleanup

```bash
az ml online-endpoint delete \
  --name amazon-review-endpoint-60301575 \
  --yes
```
