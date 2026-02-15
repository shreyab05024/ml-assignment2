Machine Learning Assignment 2
Classification Models Comparison & Deployment

1. Problem Statement
The objective of this assignment is to implement and compare multiple machine learning classification models on a real-world dataset and deploy them using Streamlit.
This project demonstrates:
Implementation of multiple supervised classification algorithms
Performance evaluation using multiple metrics
Comparative analysis of model performance
Deployment of an interactive web application using Streamlit
End-to-end ML workflow from modeling to deployment

2. Dataset Description
Dataset Used: Breast Cancer Wisconsin Diagnostic Dataset (UCI Repository)
Total Samples: 569
Total Features: 30 numerical features
Classification Type: Binary
Target Classes:
0 → Malignant
1 → Benign
Feature Details
The dataset consists of computed features from digitized images of breast mass. The features include measurements such as:
Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry
Fractal dimension
Each measurement includes mean, standard error, and worst values.
An 80/20 train-test split was used. Feature scaling was applied before training the models.

3. Models Used
The following six classification models were implemented on the same dataset:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbor (KNN)
Gaussian Naive Bayes
Random Forest (Ensemble – Bagging)
XGBoost (Ensemble – Boosting)

4. Evaluation Metrics
Each model was evaluated using:
Accuracy
AUC (Area Under ROC Curve)
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

5. Model Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.97	0.99	0.98	0.96	0.97	0.94
Decision Tree	0.94	0.93	0.95	0.92	0.93	0.88
KNN	0.96	0.98	0.97	0.95	0.96	0.92
Naive Bayes	0.95	0.97	0.96	0.93	0.94	0.90
Random Forest (Ensemble)	0.98	0.99	0.99	0.97	0.98	0.96
XGBoost (Ensemble)	0.98	0.99	0.99	0.97	0.98	0.96
(Note: Values may slightly vary depending on train-test split randomness.)

6. Observations on Model Performance
ML Model Name	Observation about Model Performance
Logistic Regression	Performed strongly due to the structured nature of the dataset. Provides good interpretability.
Decision Tree	Captured non-linear relationships but showed slight tendency to overfit.
KNN	Performed well after feature scaling; sensitive to choice of K and distance metric.
Naive Bayes	Provided competitive performance despite independence assumption. Fast and computationally efficient.
Random Forest (Ensemble)	Reduced overfitting compared to Decision Tree and achieved high overall performance.
XGBoost (Ensemble)	Achieved top performance across all evaluation metrics due to boosting and regularization.

7. Streamlit Application Features
The deployed Streamlit application includes:
CSV upload option (test dataset)
Model selection dropdown
Display of evaluation metrics
Confusion matrix
Classification report
Prediction download option
The application was successfully deployed on Streamlit Community Cloud.

8. Project Structure
ml-assignment2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    │-- train_logistic_regression.py
    │-- train_decision_tree.py
    │-- train_knn.py
    │-- train_naive_bayes.py
    │-- train_random_forest.py
    │-- train_xgboost.py
    │-- utils.py
    │-- artifacts/
         │-- *.pkl files

