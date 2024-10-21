from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from app.schemas.algorithm import Algorithm
from app.schemas.data_split_config import DataSplitConfig


class ModelBase(BaseModel):
    name: str
    hyperparameters: Optional[Dict[str, Union[List, int, str]]] = {}
    model_file: Optional[str] = None

    accuracy: Optional[float]
    f1_score: Optional[float]
    model_file: Optional[str]
    status: Optional[str]
    config_id: Optional[int]
    algorithm_id: Optional[int]

    # config: Optional[DataSplitConfig] = None
    # algorithm: Optional[Algorithm] = None

    class Config:
        orm_mode = True

class ModelCreate(ModelBase):
    pass

class ModelUpdate(ModelBase):
    name: Optional[str]
    hyperparameters: Optional[Dict[str, Union[List, int, str]]] = None
    model_file: Optional[str] = None

    accuracy: Optional[float]
    f1_score: Optional[float]
    model_file: Optional[str]
    status: Optional[str]
    config_id: Optional[int]
    algorithm_id: Optional[int]

class ModelInDB(ModelBase):
    id: int
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