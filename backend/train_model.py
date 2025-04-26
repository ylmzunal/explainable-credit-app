# train_model.py

"""
Train a credit‐scoring model on the UCI dataset and save it as app/model.pkl.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def load_data():
    # The UCI dataset is hosted as an .xls file.
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00350/"
        "default%20of%20credit%20card%20clients.xls"
    )
    # header=1 skips the title row and uses the second row as column names.
    df = pd.read_excel(url, header=1)
    # Rename the target column for clarity.
    df = df.rename(columns={'default payment next month': 'default'})
    return df

def preprocess_and_split(df):
    # 'ID' isn’t predictive, so drop it. Separate features and target.
    X = df.drop(['ID', 'default'], axis=1)
    y = df['default']
    # Stratify ensures train/test have the same default-rate distribution.
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

def train_model(X_train, y_train):
    # Pipeline chains scaling + logistic regression.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',   LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    score = pipeline.score(X_test, y_test)
    print(f"Model test accuracy: {score:.4f}")

def save_model(pipeline, path="app/model.pkl"):
    # joblib.dump writes the pipeline object to disk.
    joblib.dump(pipeline, path)
    print(f"Saved trained model to {path}")

if __name__ == "__main__":
    # 1. Load the data
    df = load_data()
    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    # 3. Train the model pipeline
    pipeline = train_model(X_train, y_train)
    # 4. Evaluate on held-out test set
    evaluate_model(pipeline, X_test, y_test)
    # 5. Save the trained pipeline for inference
    save_model(pipeline)