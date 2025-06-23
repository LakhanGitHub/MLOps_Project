from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import mlflow
import dagshub
import os

# Initialize dagshub and set mlflow for experiment tracking
dagshub.init(repo_owner='LakhanGitHub', repo_name='MLOps_Project', mlflow=True)
mlflow.set_experiment('Experiment4_2')
mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')

# Load dataset
data_path = r'E:\Data_Science\MLOps-Exp\water_potability.csv'
data = pd.read_csv(data_path)

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Function to impute missing values with mean
def fill_missing_value(df):
    for col in df.columns:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
    return df

train_processed_data = fill_missing_value(train_data)
test_processed_data = fill_missing_value(test_data)

# Separate features and target
X_train = train_processed_data.drop(columns=['Potability'], axis=1)
y_train = train_processed_data['Potability']
X_test = test_processed_data.drop(columns=['Potability'], axis=1)
y_test = test_processed_data['Potability']

# Define model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_dict = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [4, 5, 6, 10]
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_distributions=param_dict, cv=5, n_iter=50, n_jobs=-1, verbose=False)

with mlflow.start_run(run_name='Random Forest tuning') as parent_run:

    random_search.fit(X_train, y_train)  # Fit the search

    for i in range(len(random_search.cv_results_['params'])):
        
        with mlflow.start_run(run_name=f"combination_{i+1}", nested=True) as child_run:

            # Get parameter combination
            params = random_search.cv_results_['params'][i]
            
            # Train model with current params
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", random_search.cv_results_['mean_test_score'][i])
            mlflow.log_metric("test_accuracy", acc)

            # Log the trained model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Log datasets (train and test) as artifacts
            train_csv_path = "train_data.csv"
            test_csv_path = "test_data.csv"
            train_processed_data.to_csv(train_csv_path, index=False)
            test_processed_data.to_csv(test_csv_path, index=False)
            
            mlflow.log_artifact(train_csv_path, artifact_path="datasets")
            mlflow.log_artifact(test_csv_path, artifact_path="datasets")

            # Clean up temporary files
            os.remove(train_csv_path)
            os.remove(test_csv_path)

    # Log the best overall model
    mlflow.log_params(random_search.best_params_)
    mlflow.sklearn.log_model(random_search.best_estimator_, artifact_path="best_model")

    print("Best parameters found:", random_search.best_params_)
    print("Training, evaluation, model, and dataset logging completed successfully.")
