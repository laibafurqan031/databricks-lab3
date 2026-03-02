# Lab 4 Submission

### Azure Resources Successfully Created:
1. **Feature Store Entity**: `AmazonReview` (version 1)
   - Index columns: asin, reviewerID
   - Verified working

2. **Data Asset**: `amazon_electronics_features_v1_sampled` (version 1)
   - Ready for feature engineering

3. **Compute**: `Amazon-Lab` compute instance

### Pipeline Components:
All components were created and tested (code in chat history):
- split_dataset, normalize_text, review_length_features
- sentiment_features, tfidf_features, semantic_features
- merge_features

### Pipeline Run:
- Successfully ran with 1000 TF-IDF features
- Merge completed with 376,563 rows × 1,401 columns

### Screenshots Taken:
1.  Entity show command output
2.  Data asset details
3. Compute list output

##  GitHub Repository:
https://github.com/laibafurqan031/databricks-lab3/tree/lab4_feature_engineering
