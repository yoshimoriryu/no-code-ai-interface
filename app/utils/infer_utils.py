import json
import logging
import pickle

import pandas as pd
from app.models import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_file_path):
    """
    Load a model from a file.

    Parameters:
        model_file_path (str): Path to the model file.

    Returns:
        The loaded model instance.
    """
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)
    return model

def perform_inference(model, config):
    """
    Function to perform inference using a trained model and input data.

    Parameters:
        model: The trained model object.
        config: The data config object containing data details.

    Returns:
        A dictionary containing the inference results.
    """
    dataset = pd.read_csv(f"uploaded_files/{config.filename}")
    X = dataset[config.features]
    y_pred = model.predict(X)

    results = {"predictions": y_pred.tolist()}
    if config.target in dataset.columns:
        y_true = dataset[config.target]

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)

        results.update({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm.tolist()
        })

    logger.info("Inference completed successfully")
    return results
