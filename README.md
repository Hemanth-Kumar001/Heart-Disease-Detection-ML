# Heart Disease Classification with SVM

An educational binary-classification project that applies a Support Vector Classifier (SVC) to a health-feature dataset and evaluates it with Stratified 5-Fold Cross-Validation.

> IMPORTANT: This is an educational machine-learning exercise only. It is not a medical device, does not provide a clinical diagnosis, and must not be used to make healthcare decisions.

## Dataset

File: Heart_Disease_Detection.csv  
Rows: 600  
Target: Heart_Disease (0/1)

Input columns:
- Age
- Resting_BP
- Cholesterol
- Max_Heart_Rate
- Fasting_Blood_Sugar
- Chest_Pain_Type

The uploaded CSV has no missing values. It contains no direct personal identifiers.

## Model and workflow

- Support Vector Classifier with an RBF kernel
- Class weighting set to balanced
- Median imputation and standard scaling in a scikit-learn pipeline
- Stratified 5-Fold Cross-Validation
- Out-of-fold predictions for the classification report and confusion matrix

## Evaluation

The script reports fold-wise and mean +/- standard deviation for accuracy, precision, recall, and F1-score. It also saves an out-of-fold confusion matrix.

## Run locally

Install dependencies with: python -m pip install -r requirements.txt

Run: python heart_disease_detection.py

Generated output: heart_disease_oof_confusion.png

## Notes

- The target is expected to be binary and encoded as 0 and 1.
- Recall for class 1 is the proportion of positive-label cases identified in this dataset; it does not establish clinical sensitivity.
- No performance values are claimed here; run the script to generate results in your environment.
- Dataset source and redistribution terms were not provided with the uploaded file. Confirm the source/license before redistributing the CSV elsewhere.
