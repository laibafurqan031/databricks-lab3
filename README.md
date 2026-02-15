# DSAI3202 – Winter 2026
## Lab 3: Data Preprocessing in Azure 

## Overview
This lab implements an end-to-end data preprocessing pipeline using Azure Databricks and Apache Spark, following the Medallion (Lakehouse) Architecture.

The Amazon Electronics Reviews dataset was processed through:
- Bronze Layer (Raw JSON)
- Silver Layer (Cleaned Parquet)
- Gold Layer (Curated analytics-ready dataset)

---

## Lakehouse Architecture

Bronze:
- Raw JSON data stored in raw/
- No transformations

Silver:
- Cleaned and validated data
- Stored in processed/
- Parquet format

Gold:
- Curated dataset for analytics and ML
- Stored in curated/features_v1/

---

## ETL Pipeline

Three Databricks notebooks were created:

1. 01_load_and_clean_reviews  
   - Load Parquet reviews  
   - Remove missing values  
   - Enforce valid ratings (1–5)  
   - Clean review text  

2. 02_enrich_with_metadata  
   - Load product metadata (JSON)  
   - Select relevant columns (asin, title, brand, price)  
   - Perform left join with reviews  

3. 03_write_gold_features_v1  
   - Select final curated features  
   - Write dataset to Gold layer  

The notebooks were orchestrated using a Databricks Job to run sequentially.

---

## Technologies Used

- Apache Spark (distributed data processing)
- Azure Databricks (managed Spark environment)
- Azure Data Lake Storage Gen2 (data storage)
- Parquet (columnar storage format)
- JSON (raw metadata format)

---

## Gold Dataset Features

The final curated dataset includes:

- asin
- title
- brand
- price
- reviewerID
- overall
- summary
- reviewText
- helpful
- reviewTime
- review_year

This dataset is analytics-ready and ML-ready.

---

## Gold Layer Visualizations

A new notebook was created to analyze the curated dataset.

### Visualization 1: Average Rating Over Time
- Grouped by review_year
- Calculated average rating
- Line chart

Shows rating trends and customer satisfaction changes over time.

### Visualization 2: Rating Distribution
- Counted ratings (1–5)
- Bar chart

Shows overall sentiment distribution and rating skewness.

### (Optional) Brand Comparison
- Average rating per brand
- Identifies high-performing brands.

---

## Additional Enrichment Ideas

The Gold layer can be further improved by:

- Sentiment analysis on reviewText
- Helpfulness ratio feature
- Price category segmentation
- Brand-level aggregates (avg rating, review count)
- Time-based features (month, quarter)
- Review length feature

---

## Databricks Job

Job Name: lab3_data_preprocessing_job

Pipeline Order:
01_load_and_clean_reviews  
→ 02_enrich_with_metadata  
→ 03_write_gold_features_v1  

The job ensures automated, sequential execution of the ETL pipeline.

---

## Learning Outcomes

- Understanding Lakehouse architecture
- Data cleaning and validation with Spark
- Joining structured and semi-structured data
- Creating curated datasets
- Orchestrating pipelines in Databricks
- Performing basic analytics and visualization

---
