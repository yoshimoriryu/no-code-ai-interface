import json
import os
import logging

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models import Model
from app.schemas import ModelCreate, ModelUpdate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

class CRUDModel(
    CRUDBase[Model, ModelCreate, ModelUpdate]
):
    def create_model(self, db: Session, model: ModelCreate):
        logger.info('masuk create')
        db_model = Model(
            name=model.name,
            algorithm_id=model.algorithm_id,
            hyperparameters=json.dumps(model.hyperparameters),
            config_id=model.config_id
            # model_file=f"{model.name}.pkl"
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model

    def get_model(self, db: Session, model_id: int):
        return db.query(Model).filter(Model.id == model_id).first()

    def delete_model(self, db: Session, model_id: int):
        model = db.query(Model).filter(Model.id == model_id).first()
        if model:
            logger.info(model.hyperparameters)
            db.delete(model)
            model = Model(
                id=model.id,
                name=model.name,
                algorithm_id=model.algorithm_id,
                hyperparameters=json.loads(model.hyperparameters) or {},
                model_file=f"{model.name}.pkl",
                created_at=model.created_at,
                config_id=model.config_id,
                accuracy=model.accuracy,
                f1_score=model.f1_score,
                status=model.status
            )
            logger.info(type(model.hyperparameters))
            db.commit()
            return model
        return None

    def update_model(self, db: Session, model_id: int, model_update: ModelUpdate):
        db_model = db.query(Model).filter(Model.id == model_id).first()
        if db_model:
            for key, value in model_update.dict(exclude_unset=True).items():
                setattr(db_model, key, value)
            db.commit()
            db.refresh(db_model)
            return db_model
        return None

model = CRUDModel(Model)
