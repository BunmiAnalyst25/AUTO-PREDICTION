# Contents of /auto-prediction/auto-prediction/src/predict.py

import os
import joblib
import pandas as pd

def load_model(model_path):
    """Load the trained model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

def make_predictions(model, input_data):
    """Make predictions using the loaded model and input data."""
    predictions = model.predict(input_data)
    return predictions

def main(input_csv, model_path):
    """Main function to load model and make predictions on new data."""
    # Load the model
    model = load_model(model_path)

    # Load input data
    input_data = pd.read_csv(input_csv)

    # Make predictions
    predictions = make_predictions(model, input_data)

    # Output predictions
    output_df = input_data.copy()
    output_df['Predictions'] = predictions
    print(output_df)

if __name__ == "__main__":
    # Define paths
    model_path = os.path.join('..', 'models', 'pm_pipeline.joblib')
    input_csv = os.path.join('..', 'data', 'new_data.csv')  # Update with actual input data path

    # Run the main function
    main(input_csv, model_path)