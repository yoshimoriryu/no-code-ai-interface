import os

import numpy as np
import pandas as pd


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