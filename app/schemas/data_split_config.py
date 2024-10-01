from typing import Optional, Any, List
from pydantic import BaseModel
from datetime import datetime

class DataSplitConfigBase(BaseModel):
    filename: str
    train_size: float
    random_seed: Optional[int]
    features: List[str]
    target: str
    created_at: Optional[datetime]

    class Config:
        orm_mode = True

class DataSplitConfigCreate(DataSplitConfigBase):
    pass

class DataSplitConfigUpdate(BaseModel):
    filename: Optional[str]
    train_size: Optional[float]
    random_seed: Optional[int]
    features: Optional[List[str]]
    target: Optional[str]

class DataSplitConfigInDB(DataSplitConfigBase):
    class Config:
        orm_mode = True

class DataSplitConfig(DataSplitConfigInDB):
    pass