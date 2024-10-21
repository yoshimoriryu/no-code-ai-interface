import os
import sys
import time
import json

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_delay, wait_fixed

from app.database import SessionLocal
from app.models.algorithm import Algorithm  # Import the Algorithm model

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@db/mlapp')

@retry(stop=stop_after_delay(30), wait=wait_fixed(2))
def wait_for_db():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        pass

def main():
    wait_for_db()  # Wait for the database to be ready
    print("Populating the database...")

# Add the root directory of your project to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dummy data generator for algorithms
def generate_dummy_algorithm_data():
    algorithms = {
        'Random Forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2
        },
        'SVM': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        },
        'Logistic Regression': {
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 100
        },
        'K-Means': {
            'n_clusters': 8,
            'init': 'k-means++',
            'max_iter': 300
        },
        'Decision Tree': {
            'max_depth': None,
            'min_samples_split': 2,
            'criterion': 'gini'
        },
        'Gradient Boosting Classifier': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        },
        'AdaBoost Classifier': {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R'
        },
        'Naive Bayes': {
            'var_smoothing': 1e-9
        },
        'K-Nearest Neighbors': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        },
        'Linear Discriminant Analysis': {
            'solver': 'svd',
            'shrinkage': None,
            'priors': None
        },
        'Quadratic Discriminant Analysis': {
            'priors': None,
            'reg_param': 0.0
        },
        'Extra Trees Classifier': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2
        },
        'Ridge Regression': {
            'alpha': 1.0,
            'solver': 'auto',
            'max_iter': None
        },
        'Lasso Regression': {
            'alpha': 1.0,
            'selection': 'cyclic'
        },
        'ElasticNet Regression': {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'max_iter': 1000
        },
        'Isolation Forest': {
            'n_estimators': 100,
            'max_samples': 'auto'
        }
    }

    algorithm_data_list = []

    for algorithm_name, hyperparameters in algorithms.items():
        algorithm_data = {
            'name': algorithm_name,
            'description': f'Default description for {algorithm_name}',
            'default_hyperparameters': json.dumps(hyperparameters)
        }
        algorithm_data_list.append(algorithm_data)

    return algorithm_data_list

# Main function to populate the database
def populate_database(db: Session):
    # Check if any algorithms exist
    existing_algorithm = db.query(Algorithm).first()

    if existing_algorithm:
        print("Algorithms already exist in the database. Skipping population.")
        return

    for algorithm_data in generate_dummy_algorithm_data():
        algorithm = Algorithm(**algorithm_data)
        db.add(algorithm)
    db.commit()
    print('Inserted algorithms into the database.')

# Entry point
if __name__ == "__main__":
    wait_for_db()
    # Create a new session
    db: Session = SessionLocal()

    try:
        populate_database(db)
    finally:
        db.close()
