from typing import Optional, Dict, Union, List
from pydantic import BaseModel


class AlgorithmBase(BaseModel):
    name: str = None
    description: Optional[str] = None
    default_hyperparameters: Optional[Dict[str, Union[List, int, str]]] = None

    class Config:
        orm_mode = True


class AlgorithmCreate(AlgorithmBase):
    pass


class AlgorithmUpdate(AlgorithmBase):
    name: Optional[str]
    description: Optional[str] = None
    default_hyperparameters: Optional[Dict[str, Union[List, int, str]]] = None


class AlgorithmInDB(AlgorithmBase):
    id: int
    class Config:
        orm_mode = True


class Algorithm(AlgorithmInDB):
    pass
