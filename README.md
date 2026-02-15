# DSAI3202 – Winter 2026
## Lab 3: Data Preprocessing in Azure (Lakehouse Architecture)

## Overview

This lab implements an end-to-end ETL pipeline using Azure Databricks and Apache Spark following the Medallion (Lakehouse) Architecture.

The Amazon Electronics Reviews dataset is processed through:

- Bronze Layer → Raw JSON data
- Silver Layer → Cleaned and processed Parquet data
- Gold Layer → Curated analytics-ready dataset

---

## Lakehouse Architecture

Bronze:
- Raw metadata in JSON format
- Stored in raw/

Silver:
- Cleaned and validated reviews
- Stored in processed/
- Parquet format for better performance

Gold:
- Final curated dataset
- Stored in curated/features_v1/
- Ready for analytics and ML

---

## ETL Pipeline (Notebook Mapping)

1. 01_load_and_clean_reviews  
   - Load Silver Parquet data  
   - Remove null values  
   - Enforce rating range (1–5)  
   - Clean review text  

2. 02_enrich_with_metadata  
   - Load Bronze JSON metadata  
   - Select relevant columns  
   - Left join with cleaned reviews  

3. 03_write_gold_features_v1  
   - Select final features  
   - Write curated dataset to Gold layer  

The notebooks were orchestrated using a Databricks Job to ensure sequential execution.

---

## Technologies Used

Apache Spark  
- Distributed data processing engine  
- Used for filtering, joining, transforming large datasets  

Azure Databricks  
- Managed Spark platform  
- Used to create clusters, notebooks, and ETL jobs  

Azure Data Lake Storage Gen2  
- Cloud storage for Bronze, Silver, and Gold layers  
- Accessed using abfss:// paths  

Parquet  
- Columnar storage format  
- Faster reads and better compression  

JSON  
- Used for raw metadata storage  

Delta Lake (future improvement)  
- Provides ACID transactions, schema evolution, and time travel  

---

## Code Documentation

### Storage Connection

spark.conf.set(...)

Configures Spark to authenticate and access Azure Data Lake using storage account credentials.

---

### Load Parquet Data

reviews_df = spark.read.parquet(reviews_path)

Loads processed reviews into a Spark DataFrame.

---

### Data Cleaning

filter(col("asin").isNotNull() & ...)

Removes rows with missing critical fields.

filter((col("overall") >= 1) & (col("overall") <= 5))

Ensures valid rating values.

withColumn("reviewText", trim(col("reviewText")))

Cleans review text and removes short reviews.

---

### Write Cleaned Data

clean_reviews_df.write.mode("overwrite").parquet(clean_reviews_path)

Writes cleaned data back to the Silver layer.

---

### Load Metadata

metadata_df = spark.read.json(metadata_path)

Loads raw JSON metadata.

select("asin", "title", "brand", "price")

Keeps only relevant enrichment fields.

---

### Join Operation

clean_reviews_df.join(metadata_df, on="asin", how="left")

Enriches reviews with product metadata using a left join.

---

### Create Gold Dataset

select(...)

Selects final curated columns.

write.mode("overwrite").parquet(gold_path)

Writes final dataset to Gold layer.

---

## Gold Dataset Features

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

### 1. Average Rating Over Time
Grouped by review_year and calculated average rating.
Shows rating trends and customer satisfaction changes over time.

### 2. Rating Distribution
Counted ratings (1–5) and plotted bar chart.
Shows overall sentiment distribution and rating skewness.

### 3. Brand Comparison (Optional)
Average rating per brand.
Identifies high-performing brands.

---

## Additional Enrichment Opportunities

- Sentiment analysis on reviewText
- Helpfulness ratio feature
- Price segmentation (low/medium/high)
- Brand-level aggregates
- Time-based features (month, quarter)
- Review length feature

---

## Databricks Job

Job Name: lab3_data_preprocessing_job

Pipeline Order:
01_load_and_clean_reviews  
→ 02_enrich_with_metadata  
→ 03_write_gold_features_v1  

Ensures automated and sequential ETL execution.

---

## Learning Outcomes

- Understanding Lakehouse architecture
- Data cleaning and validation using Spark
- Joining structured and semi-structured data
- Creating curated datasets
- Orchestrating pipelines in Databricks
- Performing analytical visualizations
