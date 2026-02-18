# Demo Projects Repository

This repository contains projects and assignments completed during my ISB AMPBA program.
It is intended to showcase practical skills and applied learnings across statistics, optimization, and machine learning.

---

## Table of Contents

- [Project 1: Statistical Analysis — Basics of Hypothesis Testing & Understanding the p-value](./Project%2001:%20Statictical%20Analysis%20:%20Basics%20of%20Hypothesis%20Testing%20and%20undterstanding%20the%20p-value/)
- [Project 2: Optimization & Simulation — Gentle Lentil Case Study](./Project%2002:%20Optmisation%20and%20Simulation%20:%20Gentel%20Lentil%20Case%20Study/)
- [Project 3: Unsupervised Learning — Customer Behavioral Segmentation](./Project%2003:%20Unsupervised%20Learning%20:%20Customer%20Behavioural%20Segmentation/)
- [Project 4: Unsupervised Learning — Company Segmentation based on Performance](./Project%2004:%20Unsupervised%20Learning%20:%20Company%20Segmentation%20on%20the%20bases%20of%20Performance/)
- [Project 5: Big Data Management Basics](./Project%2005:%20Big%20Data%20Management%20Basics/)
- [Project 6: Causal Inference Checks](./Project%2006:%20Causal%20Inference%20checks/)
- [Project 7: Supervised Learning Basics — Decision Tree & MNIST Classification](./Project%2007:%20Supervised%20Learning%20Basics%20/)
- [Project 8: Basket Item Recommendation System](./Project%2008:%20Basket%20Item%20Reccomendation/)
- [Project 9: MLOps — AWS-based App Deployment (Practice)](./Project%2009:%20MLOps%20:%20AWS%20based%20App%20deployment%20-%20Practice/)
- [Project 10: LDA, LR, Survival Analysis & Poisson Regression on Telecom Churn](./Project%2010:%20LDA,%20LR,%20Survival%20Analysis%20and%20PoisReg/)
- [Project 11: Deep Learning Basics — Practical Use Cases](./Project%2011:%20Deep%20Learning%20Basic%20Use%20cases/)
- [Project 12: ML Supervised Learning Classifiers](./Project%2012:%20Different%20Types%20of%20Classifier/)
- [Project 13: Forecasting — Stock Price Time Series (AAPL)](./Project%2013:%20Stock%20Forecasting%20Analysis/)
- [Project 14: LLMs / RAG — Knowledge Graph–Enhanced RAG with Wikipedia + Neo4j](./Project%2014:%20Knowledge%20Graph%E2%80%93enhanced%20RAG%20system%20using%20Wikipedia%20data/)
- [Project 15: Marketing Analytics — Product Optimization, Segmentation & Ads](./Project%2015:%20Marketing%20and%20Customer%20Analytics/)
- [Project 16: Predictive Analytics (R) — Aravind Screening vs Surgery Modeling](./Project%2016:%20Predictive%20Analytics%20(R)%20-%20Aravind%20Screening%20vs%20Surgery%20Modeling/)
- [Project 16: Predictive Analytics (R) — Aravind Screening vs Surgery Modeling](./Project%2016:%20Aravind%20Screening%20Surgery%20Interaction%20Model/)
- [Project 18: Forecasting App — Florida Energy Demand Forecasting (Streamlit + Docker)](./Project%2018:%20Forecasting%20App%20%E2%80%94%20Florida%20Energy%20Demand%20Forecasting/)

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

**Overview**

Unsupervised segmentation of mall visitors based on dwell-time/engagement features to identify distinct behavioral groups.

**Key learnings**

- K-means clustering on normalized dwell-time features to segment mall visitors by floor-level engagement
- Use of the Elbow method and Silhouette score to select the number of clusters and interpret cluster profiles
- Behavioral cuts such as filtering employees/short visits, comparing Apple vs non-Apple users, and identifying high-traffic sections

---

### Project 4: Unsupervised Learning — Company Segmentation based on Performance

**Files**

- `Sectoral_segment_part_1.ipynb`
- `Sectoral_segment_part_2.ipynb`

**Overview**

Company segmentation for the pharmaceutical sector using financial ratios and clustering-style analysis to group firms based on performance/characteristics. Data was collected from Screener.in and ratios were analyzed to compare companies across size categories.

**Key learnings**

- Feature engineering and ratio-based comparisons for company performance profiling
- Segmenting companies into interpretable groups using unsupervised analysis and visualization

---

### Project 5: Big Data Management Basics

**Files**

- `Big_data_management_basics.ipynb`

**Overview**

Introductory notebook covering data ingestion and transformations using PySpark concepts (RDDs and related operations).

**Key learnings**

- Basics of big data management with PySpark and basic RDD operations and transformations

---

### Project 6: Causal Inference Checks

**Files**

- `Causal_inference_project.Rmd`
- `Causal_inference_project.html`

**Overview**

Analysis of wage data focusing on the relationship between wages, IQ, and education. Regression results suggest both IQ and education have statistically significant effects on wages. An omitted variable bias check confirms the theoretical relationship between simple and multiple regression coefficients.

**Key learnings**

- Built and interpreted simple vs multiple regression models to quantify effects and statistical significance
- Verified omitted-variable-bias relationships between coefficients using theory-backed checks

---

### Project 7: Supervised Learning Basics — Decision Tree & MNIST Classification

**Files**

- `decision_tree.ipynb`
- `tmnist.ipynb`

**Overview**

Supervised learning practice notebooks implementing multiple classification approaches (Decision Trees and classic non-parametric methods) on MNIST/tMNIST-style image data.

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

**Key learnings**

- Built recommendation logic using association rules and collaborative filtering concepts
- Framed the problem as a real-world next-item recommendation task and evaluated iteratively (competition setting)

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

**Overview**

Practice MLOps workflow to package and deploy a bank prediction Streamlit application using Docker/Docker Compose, then push the image to AWS ECR and run it on an EC2 instance (with AWS setup steps documented).

**Key learnings**

How to set up an AWS registry and deploy a Streamlit app using Docker and Docker Compose on an EC2 instance (basic guided deployment).

---

### Project 10: LDA, LR, Survival Analysis & Poisson Regression on Telecom Churn

**Files**

- `telecomm_churn.ipynb`

**Overview**

Modeling and analysis on telecom churn data using multiple statistical/ML techniques (classification and time-to-event methods) to understand drivers of churn and evaluate predictive performance.

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

---

### Project 13: Forecasting — Stock Price Time Series (AAPL)

**Files**

- `Stock_forecasting_aapl.ipynb`

**Overview**

Time series analysis and forecasting exercise on Apple Inc. (AAPL) stock prices using Yahoo Finance market data (Jan 2020–Aug 2025) and a 365-day forecast horizon.

**Key learnings**

- End-to-end forecasting workflow: data acquisition, decomposition/stationarity checks, and model evaluation
- Compared multiple forecasting approaches (ARIMA-style methods, tree-based regression, Prophet)
- Time-series cross-validation and error metrics (MAE/MSE/MAPE) for model selection

---

### Project 14: LLMs / RAG — Knowledge Graph–Enhanced RAG with Wikipedia + Neo4j

**Files**

- `kg_rag_wikipedia_neo4j.ipynb`

**Overview**

Implemented a Knowledge Graph–enhanced Retrieval-Augmented Generation (RAG) system using Wikipedia data. The pipeline loads Wikipedia articles, creates a Neo4j-backed graph + vector index, and queries via an LLM.

**Key learnings**

- Building KG + vector hybrid retrieval using Neo4j, document chunking, and embeddings
- Using LangChain components to orchestrate retrieval, graph queries, and response generation
- Handling LLM/infra concerns such as rate limits and environment configuration

---

### Project 15: Marketing Analytics — Product Optimization, Segmentation & Ads

**Files**

- `Product_optimization.ipynb`
- `Affinity_segmentation.ipynb`
- `Paid_search_bid_optimization.ipynb`
- `Display_advertising_assessment.ipynb`

**Overview**

Multi-part Marketing and Customer Analytics assignment covering product optimization, affinity-based segmentation, paid search bid optimization, and display advertising assessment.

**Key learnings**

- Translating customer preference/choice-style data into product and segmentation insights
- Cluster/segmentation analysis using engineered features and interpretable segment profiles
- Optimization under constraints for marketing spend (e.g., bid optimization)
- Evaluating advertising performance metrics and trade-offs

---

### Project 16: Predictive Analytics (R) — Aravind Screening vs Surgery Modeling

**Files**

- `Aravind_screening_surgery_modeling.Rmd`
- `Aravind_screening_surgery_modeling.html`

**Overview**

R-based predictive modeling assignment analyzing how paid and free screening volumes (including interaction effects) relate to total surgeries, using log transformations and regression modeling.

**Key learnings**

- Log-linear regression modeling and interpretation of elasticities
- Interaction effects and model selection using adjusted R-squared
- Turning regression coefficients into business-relevant scenario impacts

---
### Project 18: Forecasting App — Florida Energy Demand Forecasting (Streamlit + Docker)

**Files**

- `energy-demand-forecasting_full_project/README.md`
- `energy-demand-forecasting_full_project/app.py`
- `energy-demand-forecasting_full_project/Dockerfile`
- `energy-demand-forecasting_full_project/requirements.txt`
- `energy-demand-forecasting_full_project/src/train.py`
- `energy-demand-forecasting_full_project/src/predict.py`

**Overview**

Full mini-project to forecast Florida energy demand with a reproducible training pipeline and a Streamlit UI, including Docker packaging for deployment.

**Key learnings**

- Structuring an end-to-end forecasting project with `src/` modules (data, features, training, prediction)
- Building an interactive Streamlit interface for forecasting outputs
- Packaging and running the application locally and via Docker

 