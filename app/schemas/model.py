from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ModelBase(BaseModel):
    id: int
    name: str
    algorithm: str
    hyperparameters: Optional[Dict[str, List]] = None
    model_file: Optional[str] = None

    class Config:
        orm_mode = True

class ModelCreate(ModelBase):
    pass

class ModelUpdate(ModelBase):
    name: Optional[str]
    algorithm: Optional[str]
    hyperparameters: Optional[Dict[str, List]] = None
    model_file: Optional[str] = None

class ModelInDB(ModelBase):
    class Config:
        orm_mode = True

class Model(ModelInDB):
    pass

# class TrainData(BaseModel):
#     X_train: List[List[float]]
#     y_train: List[int]
#     X_test: List[List[float]]
#     y_test: List[int]

# class InferenceData(BaseModel):
#     data: List[List[float]]