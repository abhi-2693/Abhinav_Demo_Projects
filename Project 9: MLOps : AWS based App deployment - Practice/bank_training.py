
# The code in this file is divided into the following sections:
#
# 1. Import necessary libraries
# 2. [Function name : perform_eda] Define a function to perform exploratory data analysis (EDA):
#    - Print feature statistics
#    - Plot target variable distribution
#    - Plot duration vs subscription
#    - Plot age distribution
#    - Plot subscription rate by job type
# 3. [Function name : prepare_data] Define a function to perform data preparation:
#    - Encode categorical variables using one-hot encoding
#    - Perform feature engineering:
#      - Create interaction features
#      - Create categorical feature combinations
#      - Create time-based features
#    - Train model with GridSearchCV
# 6. [Function name : evaluate_model] Define a function to evaluate a machine learning model:
#    - Print classification report
#    - Print confusion matrix
# 7. [Function name : main] Read in the dataset, perform EDA, feature engineering, train and evaluate a model
#################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder,  OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def perform_eda(df, eda_dir="data/eda"):
    """Perform exploratory data analysis and save plots"""
    print(f"\n=== Exploratory Data Analysis ===")
    
    # Print feature statistics
    print("\nFeature Statistics:")
    print("Given features:", len(df.columns))
    print("\nFeatures summary:")
    for col in df.columns:
        print(f"\n{col} stats:")
        print(df[col].describe())

    # Create EDA directory
    os.makedirs(eda_dir, exist_ok=True)
    
    # 1. Target variable distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='y', data=df)
    plt.title("Target Variable Distribution (0 = No, 1 = Yes)")
    plt.savefig(os.path.join(eda_dir, "target_distribution.png"))
    print(f"Target variable distribution saved to {os.path.join(eda_dir, 'target_distribution.png')}")
    plt.close()
    
    # 2. Duration vs Subscription
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='y', y='duration', data=df)
    plt.title("Duration vs Subscription")
    plt.savefig(os.path.join(eda_dir, "duration_vs_y.png"))
    print(f"Duration vs Subscription saved to {os.path.join(eda_dir, 'duration_vs_y.png')}")
    plt.close()
    
    # 3. Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.savefig(os.path.join(eda_dir, "age_distribution.png")) 
    print("Age distribution saved to eda/age_distribution.png")
    plt.close()
    
    # 4. Job vs Subscription Rate
    plt.figure(figsize=(12, 8))
    job_subs = pd.crosstab(df['job'], df['y'], normalize='index')
    job_subs.plot(kind='bar', stacked=True)
    plt.title("Subscription Rate by Job Type")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "job_vs_subscription.png"))
    print("Job vs Subscription saved to eda/job_vs_subscription.png")
    plt.close()

def prepare_data(df):
    """Prepare data for modeling"""

    for col in df.columns:
        df.rename(columns={col: col.replace('.', '_').replace('-', '_')}, inplace=True)

    # First perform EDA
    perform_eda(df)

    # Replace 'unknown' values with NaN and drop them
    df.replace('unknown', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Convert month to numerical (Jan=1, Feb=2, etc.)
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    # Create quarter feature
    df['quarter'] = pd.cut(df['month_num'], bins=[0, 3, 6, 9, 12], 
                          labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80], 
                          labels=['0-20', '20-40', '40-60', '60-80'])
    df['interaction_duration_type'] = np.where(df['duration']>np.mean(df['duration']), 'Long', 'Short')
    df['campaign_type'] = np.where(df['campaign']>np.mean(df['campaign']), 'High', 'Low')
    df['last_contact_type'] = np.where(df['pdays']<0, 'Never','Prev_contacted')

    print("\nFeature Statistics:")
    print("New features created:", len(['quarter', 'age_bin', 'interaction_duration_type', 'campaign_type', 'last_contact_type']))
    print("\nNew features summary:")
    for col in ['quarter', 'age_bin', 'interaction_duration_type', 'campaign_type', 'last_contact_type']:
        print(f"\n{col} stats:")
        print(df[col].describe())

    # Target encoding
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Label encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(col)
        df[col] = le.fit_transform(df[col])
        print(df[col].describe())
    
    # Split features and target
    print(df.info())
    X = df[df.select_dtypes(exclude=['object']).columns].drop('y', axis=1)
    print(X.info())
    y = df['y']
    print(y.info())
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Simple train-test split without SMOTE
    print("Splitting data into training and testing sets...")
    
    # Identify numeric and categorical features
    numeric_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Get indices of numeric and categorical features
    numeric_indices = [X.columns.get_loc(col) for col in numeric_feats]
    categorical_indices = [X.columns.get_loc(col) for col in categorical_feats]
    
    # Numeric pipeline
    num_pipe = Pipeline([
    ('impute',    SimpleImputer(strategy='mean')),
    ('scale',     MinMaxScaler())
    ])
    
    # Categorical pipeline
    cat_pipe = Pipeline([
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Full preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_indices),
        ('cat', cat_pipe, categorical_indices)
    ], remainder='passthrough')
    
    # Feature selection (select top 20 features)
    selector = SelectKBest(score_func=f_classif, k=20)
    
    # Full preprocessing pipeline
    full_pipeline = Pipeline([
        ('prep', preprocessor),
        ('fs', selector)
    ])
    
    # Apply preprocessing to training data
    X_train_preprocessed = full_pipeline.fit_transform(X_train, y_train)
    X_test_preprocessed = full_pipeline.transform(X_test)
    
    # Save preprocessing pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(full_pipeline, "bank_pre_process.joblib")
    print("Preprocessing pipeline saved to bank_pre_process.joblib")    
    
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate metrics
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'support': report['1']['support']
    }
    
    # Print metrics
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Support: {metrics['support']}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'Confusion Matrix - {type(model).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'data/eda/confusion_matrix_{type(model).__name__}.png')
    plt.close()
    
    return metrics

def select_best_model(model_metrics):
    """Select the best model based on accuracy"""
    best_model = None
    best_acc = 0
    best_metrics = None
    
    for model_name, metrics in model_metrics.items():
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_model = model_name
            best_metrics = metrics
    
    print("\n=== Best Model Selection ===")
    print(f"Best Model: {best_model}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Metrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return best_model, best_metrics

def main():
    # 1. Load dataset
    df = pd.read_csv('data/bank-additional.csv')
    
    # 2. Basic data summary
    print("Dataset Info:")
    print(df.info())
    print("\nDistribution:")
    print(df['y'].value_counts())
    
    # 3. Prepare data for modeling (includes EDA)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 4. Model comparison
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'knn_base': KNeighborsClassifier()
    }
    
    # 5. Hyperparameter grids
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'knn_base': {'n_neighbors': list(range(3, 50, 2))}
    }
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/individual_models", exist_ok=True)
    
    # 6. Train and evaluate each model
    model_metrics = {}
    best_model = None
    
    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        
        # Hyperparameter tuning
        if model_name in param_grids:
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model and get metrics
        metrics = evaluate_model(model, X_test, y_test)
        model_metrics[model_name] = metrics
        
        # Save the model
        model_path = f"./models/individual_models/{model_name.replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to {model_path}")

        # Update dictionary with the trained model so it's fitted
        models[model_name] = model
    
    # Select the best model
    best_model, best_metrics = select_best_model(model_metrics)
    
    # Save the best (already trained) model
    best_model_obj = models[best_model]
    best_model_path_pkl = 'best_fit_model.pkl'
    with open(best_model_path_pkl, 'wb') as f:
        pickle.dump(best_model_obj, f)
    print(f"Best model saved as {best_model_path_pkl}")
    print(f"Best model: {best_model}")
    print(f"Best model metrics: {best_metrics}")

main()
