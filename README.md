# Advanced Customer Churn Prediction

## Day 55 – Mini Project

An end-to-end machine learning project for predicting customer churn using feature engineering, preprocessing pipelines, multiple classification models, hyperparameter tuning, model evaluation, and model persistence.

## 1. Project Objective

The objective of this project is to build an optimized customer churn prediction system that can identify customers who are likely to leave a service.

The project combines the machine learning techniques learned throughout the previous days into one complete workflow.

## 2. Dataset

The project uses a feature-engineered customer churn dataset containing 50 records and 10 columns.

### Features

- age
- monthly_charges
- tenure_months
- support_calls
- total_charges
- age_group
- tenure_years
- support_call_rate
- age_group_encoded

### Target

- churn

Target distribution:

- 0 - 33 customers
- 1 - 17 customers

## 3. Feature Engineering

The dataset contains engineered features including:

- total_charges
- age_group
- tenure_years
- support_call_rate
- age_group_encoded

The categorical age_group column was not directly passed to the models. Its encoded version, age_group_encoded, was used as a numerical feature.

## 4. Machine Learning Pipeline

Dataset
↓
Feature Engineering
↓
Train/Test Split
↓
Preprocessing Pipeline
↓
Multiple ML Models
↓
Hyperparameter Tuning
↓
Model Evaluation
↓
Best Model Selection
↓
Model Saving

## 5. Models Compared

The following classification algorithms were tested:

1. Random Forest
2. Gradient Boosting
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. XGBoost

Scaling was included in the pipelines for KNN and SVM.

## 6. Hyperparameter Tuning

Hyperparameter tuning was performed using GridSearchCV.

### Random Forest

Parameters tuned:

- n_estimators
- max_depth
- min_samples_split

Best parameters:

- n_estimators = 100
- max_depth = None
- min_samples_split = 2

Cross-validation F1-Score: 0.9314

### XGBoost

Parameters tuned:

- n_estimators
- max_depth
- learning_rate

Best parameters:

- n_estimators = 50
- max_depth = 2
- learning_rate = 0.05

Cross-validation F1-Score: 0.9314

## 7. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Training Time

On the test set, all evaluated models achieved:

- Accuracy = 1.00
- Precision = 1.00
- Recall = 1.00
- F1-Score = 1.00

The tuned Random Forest and tuned XGBoost also achieved 1.00 on the test set.

The final selection in the project was Random Forest.

## 8. Confusion Matrix

The final model produced:

[[7 0]
 [0 3]]

This means:

- 7 non-churn customers were correctly classified.
- 3 churn customers were correctly classified.
- 0 false positives.
- 0 false negatives.

The visualization is available in:

confusion_matrix.png

## 9. Feature Importance

The Random Forest model identified the following important features:

| Feature | Importance |
|---|---:|
| tenure_months | 0.2339 |
| total_charges | 0.2261 |
| monthly_charges | 0.2071 |
| tenure_years | 0.1246 |
| support_call_rate | 0.0997 |
| age | 0.0951 |
| support_calls | 0.0135 |
| age_group_encoded | 0.0000 |

The visualization is available in:

feature_importance.png

## 10. Best Model

The project selected:

Random Forest

Final test-set metrics:

- Accuracy: 1.00
- Precision: 1.00
- Recall: 1.00
- F1-Score: 1.00

The trained model is saved as:

best_churn_model.pkl

## 11. Project Files

churn-prediction-advanced/
│
├── churn_prediction.py
├── feature_engineered_dataset.csv
├── best_churn_model.pkl
├── model_comparison.csv
├── tuned_model_comparison.csv
├── final_evaluation.csv
├── feature_importance.csv
├── feature_importance.png
├── confusion_matrix.png
└── README.md

## 12. How to Run

Install the required packages:

python -m pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

Run the project:

python churn_prediction.py

The script trains the models, performs hyperparameter tuning, evaluates the models, generates visualizations, and saves the best model.

## 13. Evaluation Outputs

The project generates:

- model_comparison.csv
- tuned_model_comparison.csv
- final_evaluation.csv
- confusion_matrix.png
- feature_importance.csv
- feature_importance.png
- best_churn_model.pkl

## 14. Limitations

The dataset contains only 50 records, so the test results should not be treated as representative of real-world production performance.

A larger and more diverse dataset would be required to properly validate the model before production deployment.

## 15. Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction.

It combines:

- Feature engineering
- Feature scaling
- Machine learning pipelines
- Multiple classification algorithms
- Hyperparameter tuning
- Model evaluation
- Confusion matrix analysis
- Feature importance analysis
- Model persistence

The final trained Random Forest model was saved successfully and the complete project is ready for GitHub submission.