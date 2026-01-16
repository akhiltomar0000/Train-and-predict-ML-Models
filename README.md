# Train-and-predict-ML-Models
# Machine Learning Models on Titanic Dataset
Project Overview

This project demonstrates an end-to-end machine learning workflow using the famous Titanic dataset.
Multiple classification algorithms are implemented and evaluated to predict passenger survival.

The notebook focuses on:

Data preprocessing

Feature engineering

Model training

Model evaluation

Comparison of different ML algorithms

Dataset

Source: Seaborn built-in Titanic dataset

Target variable: survived

Problem type: Binary Classification

Libraries Used

numpy

pandas

seaborn

matplotlib

scikit-learn

Project Workflow
1. Data Loading

The dataset is loaded using Seaborn:

sns.load_dataset("titanic")

2. Data Cleaning & Preprocessing

Dropped irrelevant columns:

deck, embark_town, alive, class, who, adult_male

Handled missing values:

age filled with mean

Rows with missing embarked removed

Converted categorical variables using Label Encoding:

sex

embarked

Converted dataset to integer format

3. Feature Selection

Features (X): All columns except survived

Target (y): survived

4. Train-Test Split

80% training data

20% testing data

random_state = 42

5. Models Implemented

The following machine learning models were trained and evaluated:

1. Logistic Regression

Baseline classification model

Used to understand linear decision boundaries

2. K-Nearest Neighbors (KNN)

Used after feature scaling

Distance-based algorithm

3. Naive Bayes (GaussianNB)

Probabilistic classifier

Works well with independent features

4. Decision Tree Classifier

Tree-based model

Captures non-linear relationships

5. Support Vector Machine (SVM)

Kernel: RBF

Works effectively in high-dimensional space

6. Feature Scaling

Applied StandardScaler

Used for:

KNN

Decision Tree

SVM

7. Model Evaluation

Each model is evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

Precision

Recall

F1-score

Key Learnings

Machine learning models require numerical input

Categorical data must be encoded

Scaling is essential for distance-based and kernel-based models

Different models perform differently on the same dataset

Evaluation metrics are crucial beyond accuracy

Conclusion

This notebook serves as a foundational ML project covering:

Data preprocessing

Model building

Model comparison



