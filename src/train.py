# Contents of src/train.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def load_data(data_path):
    """Load the dataset from the specified path."""
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the dataset by dropping non-predictive columns and creating features."""
    # Drop identifiers that are not predictive
    for c in ['UDI', 'Product ID']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Create derived features
    if 'Rotational speed [rpm]' in df.columns and 'Torque [Nm]' in df.columns:
        df['speed_torque'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    
    if 'Process temperature [K]' in df.columns and 'Air temperature [K]' in df.columns:
        df['temp_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']
    
    if 'Tool wear [min]' in df.columns:
        df['tool_wear_bin'] = pd.cut(df['Tool wear [min]'], bins=[-1, 50, 100, 150, 200, 300], labels=['0-50', '51-100', '101-150', '151-200', '200+'])
    
    return df

def train_model(X, y):
    """Train the predictive maintenance model."""
    # Define preprocessing for numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Create and train the model pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])
    
    model.fit(X, y)
    return model

def save_model(model, model_file):
    """Save the trained model to a file."""
    joblib.dump(model, model_file)
    print('Saved pipeline to', model_file)

def main():
    # Load dataset
    data_path = r'c:\\Users\\HP\\OneDrive\\Documents\\Auto dataset\\ai4i2020.csv'
    df = load_data(data_path)

    # Preprocess data
    df = preprocess_data(df)

    # Define features and target
    target = 'Machine failure'
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    out_path = r'c:\\Users\\HP\\OneDrive\\Documents\\Auto dataset\\models'
    os.makedirs(out_path, exist_ok=True)
    model_file = os.path.join(out_path, 'pm_pipeline.joblib')
    save_model(model, model_file)

if __name__ == "__main__":
    main()