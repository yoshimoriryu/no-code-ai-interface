import json
import os

from app.crud.base import CRUDBase
from sqlalchemy.orm import Session

from app.models import Model
from app.schemas import ModelCreate, ModelUpdate

class CRUDModel(
    CRUDBase[Model, ModelCreate, ModelUpdate]
):
    def create_model(self, db: Session, model: ModelCreate):
        db_model = Model(
            name=model.name,
            algorithm=model.algorithm,
            hyperparameters=json.dumps(model.hyperparameters),
            model_file=f"{model.name}.pkl"
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model

    def get_model(self, db: Session, model_id: int):
        return db.query(Model).filter(Model.id == model_id).first()

model = CRUDModel(Model)

# def train_model(db: Session, model_id: int, data: schemas.TrainData):
#     db_model = get_model(db, model_id)
#     if not db_model:
#         return {"error": "Model not found"}

#     X_train = data.X_train
#     y_train = data.y_train
#     X_test = data.X_test
#     y_test = data.y_test

#     hyperparameters = json.loads(db_model.hyperparameters)
#     clf = RandomForestClassifier(**hyperparameters)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     report = classification_report(y_test, y_pred)
    
#     # Save the model
#     joblib.dump(clf, db_model.model_file)

#     return {"report": report}

# def infer_model(db: Session, model_id: int, input_data: schemas.InferenceData):
#     db_model = get_model(db, model_id)
#     if not db_model:
#         return {"error": "Model not found"}

#     clf = joblib.load(db_model.model_file)
#     predictions = clf.predict(input_data.data)
#     return {"predictions": predictions.tolist()}