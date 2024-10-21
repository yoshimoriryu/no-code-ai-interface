from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel


class MissingDataStrategy(str, Enum):
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MOST_FREQUENT = "fill_most_frequent"
    REMOVE_ROWS = "remove_rows"
    REPLACE_CONSTANT = "replace_constant"
    DO_NOTHING = "do_nothing"

class DataSplitConfigBase(BaseModel):
    filename: str
    train_size: float
    random_seed: Optional[int]
    features: List[str]
    target: str
    missing_data_strategy: Optional[MissingDataStrategy]
    constant_value: Optional[float] = None  # constant value if replace_constant is chosen
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
    id: int
    class Config:
        orm_mode = True

class DataSplitConfig(DataSplitConfigInDB):
    pass