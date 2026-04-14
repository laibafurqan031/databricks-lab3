# DSAI3202 – Assignment 2  
## Model Training & Automation with Azure Machine Learning

## Assignment Objective

Build an end-to-end MLOps workflow on Azure ML including training, hyperparameter tuning, model registration, online deployment, and endpoint testing. A Logistic Regression model predicts positive vs negative Amazon reviews.

---

## Dataset

Amazon electronics reviews with engineered features: SBERT embeddings, TF-IDF, sentiment scores, review length stats.

### Splits (60/15/15/10):
| Split | Rows |
|-------|------|
| Train | 225,937 |
| Validation | 56,484 |
| Test | 56,485 |
| Deployment | 37,657 |

**Label:** `1` if overall rating >= 4 (positive), else `0`.

---

## Model Choice

**Logistic Regression** – simple, fast, interpretable, works well with high-dimensional sparse features.

---

## Hyperparameter Tuning (Sweep Job)

- **Trials:** 8 (random sampling)
- **Search Space:** C (0.001–10.0), max_iter (100, 300, 500, 1000)
- **Best Run:** `frank_lettuce_3jd6g412z5_4`
- **Best Params:** C ≈ 0.1, max_iter = 500

---

## Performance (Best Model)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 85.6% | 79.7% | 79.7% |
| AUC | 0.87 | 0.69 | 0.68 |
| F1 Score | 91.4% | 88.3% | 88.3% |

**Training Runtime:** ~76 seconds

---

## 🚀 Deployment

- **Registered Model:** `amazon-review-sentiment-model:1`
- **Endpoint:** `amazon-review-endpoint-60301575`
- **Deployment:** `amazon-review-deployment`
- **Instance:** Standard_F2s_v2

**Scoring URI:** `https://amazon-review-endpoint-60301575.qatarcentral.inference.ml.azure.com/score`

---

## Endpoint Test (Deployment Split Sample)

- **Accuracy:** 70.0% (10-sample test)
- **Predictions vs Actual:** [1,1,1,1,1] vs [1,1,0,0,1]

---

## Bonus Question
"There is one thing we are doing 'not correctly' in this assignment. What is it?"

Answer: The deployment split should come from the most recent time period (review_year) to simulate real-world data drift, but random splitting was used instead.
