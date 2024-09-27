from typing import Optional, Any
from pydantic import BaseModel

class DataSplitConfigBase(BaseModel):
    filename: str
    train_size: float
    random_seed: Optional[int]
    created_at: Any

    class Config:
        orm_mode = True

class DataSplitConfigCreate(DataSplitConfigBase):
    pass

class DataSplitConfigUpdate(BaseModel):
    filename: Optional[str]
    train_size: Optional[float]
    random_seed: Optional[int]

class DataSplitConfigInDB(DataSplitConfigBase):
    class Config:
        orm_mode = True

class DataSplitConfig(DataSplitConfigInDB):
    pass