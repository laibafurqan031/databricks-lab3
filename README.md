# DSAI3202 – Lab 4
## Text Feature Engineering with Azure ML 

## Overview

This lab focuses on converting raw textual data into numerical features suitable for machine learning models. Using a curated Amazon Electronics reviews dataset, an end-to-end feature engineering workflow is implemented using Azure Machine Learning pipelines and the Azure ML Feature Store.

The final outcome of this lab is a versioned, reusable Feature Set that will be used in downstream modeling and training tasks.

---

## Objectives

- Inspect and validate the curated Gold dataset
- Create a sampled dataset suitable for large-scale text feature engineering
- Avoid data leakage through proper dataset splitting
- Implement text feature engineering using Azure ML command components
- Build and execute an Azure ML pipeline
- Register engineered features in the Azure ML Feature Store
- Ensure reproducibility using YAML-based assets and CLI workflows

---

## Dataset Description

- Dataset: Amazon Electronics Reviews (Gold layer)
- Original size: 20M+ reviews
- Sampled size: ~300,000 reviews
- Sampling seed: 42

Sampling is performed to balance computational feasibility and statistical representativeness. The sampling strategy is designed to reduce the effect of language and temporal drift.

---

## Databricks Exploration and Validation

The following validation steps were performed in Databricks:

- Verified number of rows and columns
- Inspected schema and column data types
- Checked for missing or malformed values
- Confirmed review text columns are strings
- Confirmed rating values are numeric

### Visualizations

- Rating distribution  
  Used to understand class balance and rating skew.

- Review length distribution  
  Used to identify outliers and inform text preprocessing decisions.

These visualizations help ensure the dataset is suitable for feature engineering.

---

## Sampled Dataset Creation

A sampled version of the Gold dataset is created and written back to the curated layer as:

curated/features_v1_sampled/

This dataset is used as input to the Azure ML feature engineering pipeline.  
The original dataset remains unchanged.

---

## Azure ML Setup

### Git Branch
All work is completed on the following branch:
lab4_feature_engineering

### Azure ML Configuration

- Azure ML CLI installed and updated
- Logged in using az login
- Default resource group and workspace configured
- Azure ML compute cluster created

---

## Datastore and Data Asset Registration

- Azure Data Lake access configured using a blob datastore
- Sampled dataset registered as an Azure ML Data Asset:
  amazon_electronics_features_v1_sampled@1

This enables Azure ML pipelines to consume the data.

---

## Feature Store Entity

An Azure ML Feature Store entity is created with the following keys:

- asin
- reviewerID

This entity uniquely identifies each review record and is used to join features.

---

## Feature Engineering Components

Each feature is implemented as an Azure ML command component.

### Implemented Components

- Dataset split (train / validation / test)
- Text normalization
- Review length features
- Sentiment features
- TF-IDF features
- Semantic embedding features (SBERT)
- Feature merging

Each component:
- Performs one clearly defined task
- Is implemented in Python
- Is registered using a component YAML file
- Runs on Azure ML compute

---

## Data Leakage Prevention

The dataset is split before fitting any feature extraction logic.  
TF-IDF vectorizers and embedding models are fit only on the training split and then applied to validation and test splits.

---

## Azure ML Pipeline

The pipeline performs the following steps:

1. Load sampled dataset
2. Split into train, validation, and test sets
3. Normalize text for each split
4. Extract text-based features
5. Merge all features into a single dataset

The pipeline runs entirely on Azure ML compute and is defined declaratively using YAML.

---

## Feature Set Registration

After the pipeline completes successfully:

- The merged feature dataset is located from pipeline outputs
- Feature specification files define schema and data types
- A Feature Set is registered in the Azure ML Feature Store
- Features are versioned and reusable

---

## Deliverables

- Azure ML components (components/)
- Azure ML pipeline definition (pipelines/)
- Feature Store assets (feature_store/)
- Databricks notebooks for data inspection and visualization
- Successful Azure ML pipeline run
- Registered Feature Set in Azure ML Feature Store
- This README file

---

## Notes

- Secrets and storage keys are not committed to GitHub
- All workflows are reproducible and CLI-driven
- Features will be reused in later modeling labs
- Due to the ephemeral Azure Cloud Shell environment, local code files were lost before being pushed to GitHub; however, all Azure ML resources and pipeline runs were successfully completed.
- Screenshots of the successful pipeline run, registered Feature Store entity, data asset, and compute instance status are included in this repository as evidence.

---
