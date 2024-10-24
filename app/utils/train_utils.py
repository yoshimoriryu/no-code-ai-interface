import json
import logging
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# A dictionary of algorithms to easily map model names to scikit-learn models
MODEL_ALGORITHMS = {
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "Logistic Regression": LogisticRegression,
    "K-Means": KMeans,
    "Decision Tree": DecisionTreeClassifier,
    "Gradient Boosting Classifier": GradientBoostingClassifier,
    "AdaBoost Classifier": AdaBoostClassifier,
    "Naive Bayes": GaussianNB,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis,
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis,
    "Extra Trees Classifier": ExtraTreesClassifier,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "ElasticNet Regression": ElasticNet,
    "Isolation Forest": IsolationForest,
}

def perform_training(model, config):
    """
    Function to perform training based on the selected model and config.
    
    Parameters:
        model: The model object containing algorithm, hyperparameters, and more.
        config: The data config object containing data details (train size, etc.).

    Returns:
        A dictionary containing the training results.
    """
    # Load dataset from the config (assuming it's a CSV file path)
    dataset = pd.read_csv(f"uploaded_files/{config.filename}")

    # Split dataset into features and target
    X = dataset[config.features]
    y = dataset[config.target]

    # Handle any missing data strategy
    if config.missing_data_strategy == "mean":
        X.fillna(X.mean(), inplace=True)
    elif config.missing_data_strategy == "median":
        X.fillna(X.median(), inplace=True)
    elif config.missing_data_strategy == "most_frequent":
        X.fillna(X.mode().iloc[0], inplace=True)
    elif config.missing_data_strategy == "constant" and config.constant_value is not None:
        X.fillna(config.constant_value, inplace=True)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.train_size, random_state=config.random_seed
    )

    # Fetch the model algorithm class from the dictionary
    ModelClass = MODEL_ALGORITHMS.get(model.algorithm.name)
    if not ModelClass:
        raise ValueError(f"Unsupported model algorithm: {model.algorithm}")

    # Instantiate the model with hyperparameters (if any are provided)
    hyperparams = json.loads(model.hyperparameters)
    model_instance = ModelClass(**hyperparams)
    logger.info('Initializing training')
    default_model = RandomForestClassifier()
    logger.info(f"Default hyperparameters: {default_model.get_params()}")
    logger.info('Model Instance:')
    logger.info(model_instance)
    logger.info('Model Hyperparams: ')
    logger.info(model_instance.get_params())

    # Train the model
    model_instance.fit(X_train, y_train)

    # Make predictions
    y_pred = model_instance.predict(X_test)

    # Calculate performance metrics (e.g., accuracy and F1 score)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # Optionally save the trained model to disk
    model_path = f"trained_models/{model.name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_instance, f)

    # Return results of the training process
    return {
        "model_id": model.id,
        "config_id": config.id,
        "model_name": model.name,
        "algorithm": model.algorithm,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "model_file": f"{model.name}.pkl",
        "status": "Training completed successfully"
    }
