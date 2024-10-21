import os

import numpy as np
import pandas as pd
import pickle


def clean_dict(data):
    """Cleans a list of dictionaries by replacing NaN, inf, and -inf with None."""
    for row in data:
        for key, value in row.items():
            if pd.isna(value) or (isinstance(value, float) and (np.isinf(value) or np.isnan(value))):
                row[key] = None
    return data

# Load existing CSV files into storage at startup
def load_existing_csv_files(csv_storage):
    uploaded_dir = "uploaded_files"
    if os.path.exists(uploaded_dir):
        for filename in os.listdir(uploaded_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(uploaded_dir, filename)
                csv_storage[filename] = pd.read_csv(file_path)

def extract_metadata_from_model(pkl_file):
    # Load the model from the pickle file
    model = pickle.load(pkl_file)

    # Extract basic metadata (example using scikit-learn models)
    metadata = {}

    # Extract the algorithm used (for scikit-learn models)
    if hasattr(model, '__class__'):
        metadata['algorithm'] = model.__class__.__name__  # e.g., "RandomForestClassifier"

    # Extract hyperparameters
    if hasattr(model, 'get_params'):
        metadata['hyperparameters'] = model.get_params()  # Get hyperparameters as a dictionary

    # For neural networks (e.g., Keras), you might want to extract the architecture, layers, etc.
    if hasattr(model, 'summary'):
        # Keras models have a summary function that shows architecture
        metadata['architecture'] = model.summary()

    return metadata