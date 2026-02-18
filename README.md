# Demo Projects Repository 
This is a Repository of the Projects and Assignments I have done during my ISB AMPBA Program.
This repository is solely for the purpose of showcasing my practical skills and knowledge gained throughout the course.

## Project List

### Project 1: Statictical Analysis : Basics of Hypothesis Testing and undterstanding the p-value  
 File Name : 
 - Statistical_analysis_Hypothesis Testing_basics.R

 In this I was given sample data and we have to perform the EDA, hypothesis testing on the data and examine the statistical significance via the p-value and test statistics. Key Learning from this project where 
 - how to setup a Null Hypothesis and Alternate Hypothesis 
 - Perform the hypothesis testing and examine the statistical significance via the p-value and test statistics
 
### Project 2: Optmisation and Simulation : Gentel Lentil Case Study 
 File Name : 
 - Gentel_Lentil_Case_Study_Optimisation_and_Simulation.xlsx
 
 Analyze the "The Gentle Lentil" case to compute expected monthly salary, its standard deviation, and the impact of a partnership—then compare outcomes against his consulting offer and running the restaurant solo.

 Key learnings from this project where
 - Application of Optimization & Simulation Techniques – Using probability distributions, expected value, and variance to evaluate uncertain salary outcomes and compare different career options.
 - Decision-Making under Uncertainty – Understanding how risk (measured by standard deviation) affects real-world business choices beyond just average returns.
 - Scenario & Sensitivity Analysis – Evaluating different strategic options (consulting job, independent ownership, partnership) and testing how assumptions impact outcomes.
 - Integration of Quantitative & Qualitative Insights – Combining numerical results with managerial judgment to arrive at a well-supported recommendation.

### Project 3: Unsupervised Learning : Customer Behavioural Segmentation 
 File Name : 
 - Mall_of_america.ipynb

 Key learnings from this project where
 - K-means clustering on normalized dwell-time features to segment mall visitors by floor-level engagement.
 - Use of Elbow method and Silhouette score to select the number of clusters and interpret cluster profiles.
 - Behavioral cuts such as filtering employees/short visits, comparing Apple vs non-Apple users, and identifying high-traffic sections.

### Project 4: Unsupervised Learning : Company Segmentation on the bases of Performance
 File Name : 
 - Sectoral_segment_part_1 
 - Sectoral_segment_part_2

### Project 5: Big Data Management Basics
 File Name : 
 - Big_data_management_basics.ipynb

 Key learnings from this project where
 - the basics of Big Data Management PySpark and basiuc RDD operations and transformations

### Project 6: Causal Inference checks
 File Name : 
 - Causal_inference_project.Rmd 
 - Causal_inference_project.html

 This report presents the analysis of wage data, focusing on the relationship between wages, IQ, and education. Our finding were that regression results suggest that both IQ and education have statistically significant effects on wages. The omitted variable bias check confirms the theoretical relationship between the simple and multiple regression coefficients.

### Project 7: Supervised Learning Basics : Decision Tree and MNIST Classification
 File Name : 
 - decision_tree.ipynb 
 - tmnist.ipynb

 Key learnings from this project where
 - Application of Decision Tree for classification and regression tasks
 - Application of Parzen Window for classification of Mnist Data
 - Application of KNN for classification of Mnist Data
 - Application of Perceptron for classification of Mnist Data

### Project 8: Basket Item Recommendation System
File Name : 
 - Rec_sys.ipynb

 This scriot recommend the items to the customer based on the basket items (Did you forget to buy?) and there past purchase behavior using association rule and collaborative filtering.
 This was part of a Kaggle compition. Link - https://www.kaggle.com/competitions/did-you-forget-b-25/leaderboard (Group 11)

### Project 9: MLOps : AWS based App deployment - Practice
File Name :
 - api.py
 - bank_training.py
 - streamlite_bank.py
 - Dockerfile
 - docker-compose.yaml
 - requirements.txt
 - README.md

Key learing was how setup AWS registery and deploy the streamlit app using docker and docker compose ina EC2 Instance running in AWS. This was Basic deployment coveing the steps with guidance from out instructor on the steps and code. 

### Project 10: LDA, LR, Survival Analysis and PoisReg on the Telecomm Churn Data
File Name : telecomm_churn.ipynb

Key learnings from this project where
 - Application of LDA for classification od the Churn Telecomm customer
 - Application of LR for classification of the Churn Telecomm customer
 - Application of Survival Analysis for classification of the Churn Telecomm customer
- Application of Poisson Regression for classification of the Churn Telecomm customer

### Project 11: Deep Learning Basics : Practical Use Cases
 File Name :
 - Understanding Neural Network Forward Propagation and Backpropagation.ipynb
 - Predicting Customer Churn using Neural Networks.ipynb
 - Spam Classification using Recurrent Neural Networks.ipynb
 - Image Classification using Transfer Learning.ipynb

 A set of hands-on deep learning notebooks covering fundamentals (forward/backprop) and common applied patterns (classification with dense networks, RNNs, and transfer learning).

 Key learnings from this project where: 
 - Built intuition for forward propagation, loss gradients, and backprop parameter updates
 - Trained neural networks for churn prediction and spam detection
 - Applied sequence modeling with RNNs for text
 - Used transfer learning with pretrained CNNs (fine-tuning + augmentation) for image classification
 - Learned to diagnose overfitting via validation performance.

### Project 12: ML Supervised Learning Classifiers
 File Name : 
 - Different Types of Classifier.ipynb
 
 Supervised learning assignment benchmarking multiple classifiers on a character image dataset (tMNIST-style). The notebook covers:
 - MLP Neural Network (`MLPClassifier`) with multiple hidden-layer architectures
 - Support Vector Machines (`SVC`) with linear, polynomial, and RBF kernels
 - Random Forest (`RandomForestClassifier`) with depth and number-of-trees sweeps
 - Pair-wise linear SVM using Fisher-score-based top feature selection
 
 Preprocessing + evaluation includes removing all-zero columns, label encoding, normalization to [0,1], a stratified 70/30 train-test split, and accuracy-based comparison.

 Key learnings from this project where: 
 - Ran a structured classifier benchmark by cleaning features (zero-variance removal), encoding labels, scaling inputs, and using stratified splits for fair evaluation.
 - Compared MLP, SVM (linear/polynomial/RBF), and Random Forest while tuning model complexity (parameters, support vectors, depth/trees).
 - Implemented Fisher-score feature selection for pairwise SVMs to improve separability.
