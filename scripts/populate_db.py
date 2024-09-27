'''
currently, to populate, go inside container
    docker exec -it ml-app-app-1 bash
and then
    PYTHONPATH=. python3 scripts/populate_db.py
'''

import random

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.model import Model

import sys
import os

# Add the root directory of your project to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dummy data generator
def generate_dummy_model_data(num_entries: int):
    algorithms = ['Random Forest', 'SVM', 'Logistic Regression', 'K-Means', 'Decision Tree']
    
    for _ in range(num_entries):
        model_data = {
            'name': f'Model_{random.randint(1, 1000)}',
            'hyperparameters': {
                'param1': random.uniform(0.1, 1.0),
                'param2': random.randint(1, 10),
            },
            'algorithm': random.choice(algorithms),
            'model_file': f'model_{random.randint(1, 100)}.pkl',
        }
        yield model_data

# Main function to populate the database
def populate_database(db: Session, num_entries: int):
    for model_data in generate_dummy_model_data(num_entries):
        model = Model(**model_data)
        db.add(model)
    db.commit()
    print(f'Inserted {num_entries} dummy models into the database.')

# Entry point
if __name__ == "__main__":
    
    # Create a new session
    db: Session = SessionLocal()
    
    try:
        populate_database(db, num_entries=10)  # Change the number as needed
    finally:
        db.close()
