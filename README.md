# Demo Projects Repository

This repository contains projects and assignments completed during my ISB AMPBA program.
It is intended to showcase practical skills and applied learnings across statistics, optimization, and machine learning.

---

## Table of Contents

- [Projects](#projects)
  - [Project 1: Statistical Analysis — Basics of Hypothesis Testing & Understanding the p-value](#project-1-statistical-analysis--basics-of-hypothesis-testing--understanding-the-p-value)
  - [Project 2: Optimization & Simulation — Gentle Lentil Case Study](#project-2-optimization--simulation--gentle-lentil-case-study)
  - [Project 3: Unsupervised Learning — Customer Behavioral Segmentation](#project-3-unsupervised-learning--customer-behavioral-segmentation)
  - [Project 4: Unsupervised Learning — Company Segmentation based on Performance](#project-4-unsupervised-learning--company-segmentation-based-on-performance)
  - [Project 5: Big Data Management Basics](#project-5-big-data-management-basics)
  - [Project 6: Causal Inference Checks](#project-6-causal-inference-checks)
  - [Project 7: Supervised Learning Basics — Decision Tree & MNIST Classification](#project-7-supervised-learning-basics--decision-tree--mnist-classification)
  - [Project 8: Basket Item Recommendation System](#project-8-basket-item-recommendation-system)
  - [Project 9: MLOps — AWS-based App Deployment (Practice)](#project-9-mlops--aws-based-app-deployment-practice)
  - [Project 10: LDA, LR, Survival Analysis & Poisson Regression on Telecom Churn](#project-10-lda-lr-survival-analysis--poisson-regression-on-telecom-churn)
  - [Project 11: Deep Learning Basics — Practical Use Cases](#project-11-deep-learning-basics--practical-use-cases)
  - [Project 12: ML Supervised Learning Classifiers](#project-12-ml-supervised-learning-classifiers)

---

## Projects

### Project 1: Statistical Analysis — Basics of Hypothesis Testing & Understanding the p-value

**Files**

- `Statistical_analysis_Hypothesis Testing_basics.R`

**Overview**

Given sample data, the task was to perform EDA and hypothesis testing, then examine statistical significance via the p-value and test statistics.

**Key learnings**

- How to set up a Null Hypothesis and Alternate Hypothesis
- Perform hypothesis testing and examine statistical significance via the p-value and test statistics

---

### Project 2: Optimization & Simulation — Gentle Lentil Case Study

**Files**

- `Gentel_Lentil_Case_Study_Optimisation_and_Simulation.xlsx`

**Overview**

Analyze "The Gentle Lentil" case to compute expected monthly salary, its standard deviation, and the impact of a partnership—then compare outcomes against a consulting offer and running the restaurant solo.

**Key learnings**

- Application of optimization & simulation techniques (probability distributions, expected value, variance)
- Decision-making under uncertainty (risk measured by standard deviation)
- Scenario & sensitivity analysis across strategic options
- Integration of quantitative results with managerial judgment

---

### Project 3: Unsupervised Learning — Customer Behavioral Segmentation

**Files**

- `Mall_of_america.ipynb`

**Key learnings**

- K-means clustering on normalized dwell-time features to segment mall visitors by floor-level engagement
- Use of the Elbow method and Silhouette score to select the number of clusters and interpret cluster profiles
- Behavioral cuts such as filtering employees/short visits, comparing Apple vs non-Apple users, and identifying high-traffic sections

---

### Project 4: Unsupervised Learning — Company Segmentation based on Performance

**Files**

- `Sectoral_segment_part_1`
- `Sectoral_segment_part_2`

---

### Project 5: Big Data Management Basics

**Files**

- `Big_data_management_basics.ipynb`

**Key learnings**

- Basics of big data management with PySpark and basic RDD operations and transformations

---

### Project 6: Causal Inference Checks

**Files**

- `Causal_inference_project.Rmd`
- `Causal_inference_project.html`

**Overview**

Analysis of wage data focusing on the relationship between wages, IQ, and education. Regression results suggest both IQ and education have statistically significant effects on wages. An omitted variable bias check confirms the theoretical relationship between simple and multiple regression coefficients.

---

### Project 7: Supervised Learning Basics — Decision Tree & MNIST Classification

**Files**

- `decision_tree.ipynb`
- `tmnist.ipynb`

**Key learnings**

- Application of Decision Trees for classification and regression tasks
- Application of Parzen Window for classification of MNIST data
- Application of KNN for classification of MNIST data
- Application of Perceptron for classification of MNIST data

---

### Project 8: Basket Item Recommendation System

**Files**

- `Rec_sys.ipynb`

**Overview**

Recommend items to a customer based on basket items ("Did you forget to buy?") and past purchase behavior using association rules and collaborative filtering.

This was part of a Kaggle competition:
https://www.kaggle.com/competitions/did-you-forget-b-25/leaderboard (Group 11)

---

### Project 9: MLOps — AWS-based App Deployment (Practice)

**Files**

- `api.py`
- `bank_training.py`
- `streamlite_bank.py`
- `Dockerfile`
- `docker-compose.yaml`
- `requirements.txt`
- `README.md`

**Key learnings**

How to set up an AWS registry and deploy a Streamlit app using Docker and Docker Compose on an EC2 instance (basic guided deployment).

---

### Project 10: LDA, LR, Survival Analysis & Poisson Regression on Telecom Churn

**Files**

- `telecomm_churn.ipynb`

**Key learnings**

- Application of LDA for classification of telecom churn customers
- Application of logistic regression for classification of telecom churn customers
- Application of survival analysis for classification of telecom churn customers
- Application of Poisson regression for classification of telecom churn customers

---

### Project 11: Deep Learning Basics — Practical Use Cases

**Files**

- `Understanding Neural Network Forward Propagation and Backpropagation.ipynb`
- `Predicting Customer Churn using Neural Networks.ipynb`
- `Spam Classification using Recurrent Neural Networks.ipynb`
- `Image Classification using Transfer Learning.ipynb`

**Overview**

A set of hands-on deep learning notebooks covering fundamentals (forward/backprop) and common applied patterns (classification with dense networks, RNNs, and transfer learning).

**Key learnings**

- Built intuition for forward propagation, loss gradients, and backprop parameter updates
- Trained neural networks for churn prediction and spam detection
- Applied sequence modeling with RNNs for text
- Used transfer learning with pretrained CNNs (fine-tuning + augmentation) for image classification
- Learned to diagnose overfitting via validation performance

---

### Project 12: ML Supervised Learning Classifiers

**Files**

- `Different Types of Classifier.ipynb`

**Overview**

Supervised learning assignment benchmarking multiple classifiers on a character image dataset (tMNIST-style). The notebook covers:

- MLP Neural Network (`MLPClassifier`) with multiple hidden-layer architectures
- Support Vector Machines (`SVC`) with linear, polynomial, and RBF kernels
- Random Forest (`RandomForestClassifier`) with depth and number-of-trees sweeps
- Pair-wise linear SVM using Fisher-score-based top feature selection

Preprocessing and evaluation includes removing all-zero columns, label encoding, normalization to [0, 1], a stratified 70/30 train-test split, and accuracy-based comparison.

**Key learnings**

- Structured classifier benchmarking with consistent preprocessing and stratified splits
- Comparison of MLP, SVM (linear/polynomial/RBF), and Random Forest while tuning model complexity
- Fisher-score feature selection for pairwise SVMs to improve separability
 