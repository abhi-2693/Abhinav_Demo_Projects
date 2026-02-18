# Project 12: ML Supervised Learning Classifiers — Benchmarking

## Situation / Objective
When multiple classifier families are viable, you need a structured benchmark to compare performance fairly under consistent preprocessing. The objective is to benchmark common classifiers on a tMNIST-style character image dataset.

## Task
- Clean and preprocess the dataset.
- Train multiple classifiers and tune key hyperparameters.
- Compare models using a consistent evaluation setup.

## Actions
- Implemented preprocessing:
  - Removed all-zero columns (zero variance)
  - Encoded labels
  - Normalized features to the `[0, 1]` range
  - Used a stratified 70/30 train-test split
- Benchmarked:
  - MLP (`MLPClassifier`) across hidden-layer configurations
  - SVM (`SVC`) with linear / polynomial / RBF kernels
  - Random Forest with depth and number-of-trees sweeps
  - Pairwise linear SVM using Fisher-score feature selection

## Results / Summary
- Produced a comparable benchmark across model families with controlled preprocessing.
- Identified how model complexity and feature selection affect separability and accuracy.

## Repository contents
- `Different Types of Classifier.ipynb`
