# VITyarthi-project

# CS_Project

# Diabetes Prediction Using K-Means Clustering

This project demonstrates unsupervised learning for predicting diabetes using the K-Means clustering algorithm. The aim is to classify individuals as diabetic or non-diabetic based on health parameters from a real-world dataset.

---

## Table of Contents
- Project Overview
- Dataset Description
- Technologies Used
- Data Preprocessing
- Exploratory Data Analysis
- Model Building
- Evaluation Metrics
- Results
- How to Run
- Future Improvements

---

## Project Overview

The project applies K-Means clustering to segment people into diabetic and non-diabetic groups. Data preprocessing, feature selection, visualization, and evaluation metrics are implemented to analyze model effectiveness.

---

## Dataset Description

Attributes in the dataset:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (1=Diabetes, 0=No Diabetes)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib, Seaborn
- scikit-learn

---

## Data Preprocessing

- Checked data shape and types, handled missing values by mean imputation.
- Applied MinMaxScaler to normalize feature ranges.
- Optionally encoded categorical data, if present.

---

## Exploratory Data Analysis

- Visualized data distributions with count plots, heatmaps, and pairplots.
- Explored feature correlations and outcome distributions.

---

## Model Building

- Selected features: Glucose, Insulin, BMI 
- Used train/test split (80:20) stratified on outcome.
- K-Means clustering with 2 clusters, model fit on training data.
- Output cluster centers.

---

## Evaluation Metrics

- Classification report: precision, recall, and F1-score based on true outcomes.
- Confusion matrix with heatmap visualization.

---

## Results

- Demonstrated separation between diabetic and non-diabetic clusters.
- Model produced example predictions from scaled input.

---
