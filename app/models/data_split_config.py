import os
from datetime import datetime
from enum import Enum as PyEnum

import pytz
from sqlalchemy import Column, DateTime, Enum, Float, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import JSON

from app.db.base_class import Base

UTC7 = pytz.timezone(os.getenv("SQLALCHEMY_DATABASE_CONNECT_TIMEZONE","Asia/Jakarta"))

def get_current_time():
    return datetime.now(UTC7)

missing_data_strategy_enum = Enum(
    'fill_mean', 
    'fill_median', 
    'fill_most_frequent', 
    'remove_rows', 
    'replace_constant',
    'do_nothing' ,
    name='missingdatastrategy'
)

class DataSplitConfig(Base):
    __tablename__ = 'data_split_configs'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    train_size = Column(Float)
    random_seed = Column(Integer)
    features = Column(JSON)
    target = Column(String)
    constant_value = Column(Float)
    missing_data_strategy = Column(missing_data_strategy_enum)
    created_at = Column(DateTime, default=get_current_time)

    models = relationship("Model", back_populates="config")