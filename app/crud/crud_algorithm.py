import json
import logging

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.algorithm import Algorithm  # Import your Algorithm model
from app.schemas.algorithm import AlgorithmCreate, AlgorithmUpdate  # Import your Pydantic schemas for Algorithm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

class CRUDAlgorithm(
    CRUDBase[Algorithm, AlgorithmCreate, AlgorithmUpdate]
):
    def create_algorithm(self, db: Session, algorithm: AlgorithmCreate):
        logger.info('Creating new algorithm')
        db_algorithm = Algorithm(
            name=algorithm.name,
            description=algorithm.description,
            default_hyperparameters=json.dumps(algorithm.default_hyperparameters)  # Store as JSON
        )
        db.add(db_algorithm)
        db.commit()
        db.refresh(db_algorithm)
        return db_algorithm

    def get_algorithm(self, db: Session, algorithm_id: int):
        return db.query(Algorithm).filter(Algorithm.id == algorithm_id).first()

    def delete_algorithm(self, db: Session, algorithm_id: int):
        algorithm = db.query(Algorithm).filter(Algorithm.id == algorithm_id).first()
        if algorithm:
            logger.info(f"Deleting algorithm: {algorithm.name}")
            db.delete(algorithm)
            db.commit()
            return algorithm
        return None

    def update_algorithm(self, db: Session, algorithm_id: int, algorithm_update: AlgorithmUpdate):
        db_algorithm = db.query(Algorithm).filter(Algorithm.id == algorithm_id).first()
        if db_algorithm:
            for key, value in algorithm_update.dict(exclude_unset=True).items():
                setattr(db_algorithm, key, value)
            db.commit()
            db.refresh(db_algorithm)
            return db_algorithm
        return None

# Instantiate the CRUD object for use in other parts of the app
algorithm = CRUDAlgorithm(Algorithm)
