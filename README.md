Machine Learning Assignment 2
Classification Models Comparison & Deployment
1. Problem Statement
The objective of this assignment was to build and compare multiple supervised machine learning classification models on a real-world dataset and deploy them using Streamlit.
Rather than focusing only on model accuracy, this project emphasizes:

Comparing multiple classification algorithms on the same dataset
Evaluating models using diverse performance metrics
Understanding strengths and weaknesses of different approaches
Deploying an interactive ML application
This project simulates a real-world ML workflow — from modeling to deployment.

2. Dataset Description
Dataset Used: Breast Cancer Wisconsin Diagnostic Dataset (UCI Repository)
I selected this dataset because:
It satisfies the assignment constraints (≥12 features, ≥500 samples)
It is a well-known medical classification problem
It allows meaningful comparison between linear and ensemble models
Dataset Summary
Total Samples: 569
Total Features: 30 numerical features
Classification Type: Binary
Target Classes:
0 → Malignant
1 → Benign
Feature Details
The features represent computed characteristics of cell nuclei present in digitized images of breast mass, including:
Radius, texture, perimeter, area
Smoothness, compactness, concavity
Symmetry, fractal dimension
Mean, standard error, and worst values of each measurement
Since features are continuous and vary in scale, feature scaling was applied before training distance-based models like KNN and Logistic Regression.

3. Machine Learning Models Implemented
All six models were trained and tested on the same train-test split to ensure fair comparison.
The following classification models were implemented:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest (Ensemble – Bagging)
XGBoost (Ensemble – Boosting)

4. Evaluation Metrics Used
To evaluate performance more comprehensively, I calculated:
Accuracy
AUC (Area Under ROC Curve)
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
MCC was specifically included because it provides a balanced evaluation even when class distributions are uneven.

5. Model Performance Comparison
ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.97	0.99	0.98	0.96	0.97	0.94
Decision Tree	0.94	0.93	0.95	0.92	0.93	0.88
KNN	0.96	0.98	0.97	0.95	0.96	0.92
Naive Bayes	0.95	0.97	0.96	0.93	0.94	0.90
Random Forest	0.98	0.99	0.99	0.97	0.98	0.96
XGBoost	0.98	0.99	0.99	0.97	0.98	0.96
(Note: Minor variations may occur due to randomness in train-test split.)

6. Observations & Analysis
Below are my observations after comparing model performance:
🔸 Logistic Regression
Performed very strongly. The dataset appears to have reasonably separable classes, which explains why a linear model works well. It also offers interpretability advantages.
🔸 Decision Tree
Captured non-linear relationships but showed slight overfitting compared to ensemble models. Performance was good but less stable.
🔸 KNN
Delivered strong performance after scaling. However, it is computationally heavier during prediction and sensitive to choice of K.
🔸 Naive Bayes
Despite its strong independence assumption, it performed surprisingly well. It is simple and fast, making it useful as a baseline model.
🔸 Random Forest
Improved significantly over a single Decision Tree. Reduced overfitting and achieved higher MCC and F1 scores.
🔸 XGBoost
Achieved top performance along with Random Forest. Boosting helps reduce bias and improves predictive power. It provided the best overall balance of metrics.

7. Streamlit Application Features
The deployed Streamlit app includes:
✔ CSV upload option (test dataset)
✔ Model selection dropdown
✔ Display of selected model predictions
✔ Confusion matrix
✔ Classification report
✔ Evaluation metrics display

The application allows users to interactively test different models without modifying code.

8. Project Structure
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
     │-- logistic_regression.pkl
     │-- decision_tree.pkl
     │-- knn.pkl
     │-- naive_bayes.pkl
     │-- random_forest.pkl
     │-- xgboost.pkl
   
9. Deployment
The project was deployed on Streamlit Community Cloud by:
Uploading the repository to GitHub
Linking it to Streamlit Cloud
Selecting app.py as the main file
Ensuring all dependencies were included in requirements.txt
The deployed app runs successfully without dependency errors.

10. Conclusion
This project demonstrates the comparative strengths of different classification algorithms.
Key takeaways:

Linear models can perform competitively on structured datasets.
Ensemble methods (Random Forest & XGBoost) consistently deliver superior performance.
Model evaluation should go beyond accuracy — MCC and AUC provide deeper insight.
Deployment adds practical value to ML projects.
Among all models tested, Random Forest and XGBoost provided the most reliable performance across all evaluation metrics.


