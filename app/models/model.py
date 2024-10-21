import os
from datetime import datetime

import pytz
from sqlalchemy import Boolean, Column, DateTime, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import JSON

from app.db.base_class import Base

UTC7 = pytz.timezone(os.getenv("SQLALCHEMY_DATABASE_CONNECT_TIMEZONE","Asia/Jakarta"))

def get_current_time():
    return datetime.now(UTC7)

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True)
    hyperparameters = Column(JSON)
    model_file = Column(String)
    created_at = Column(DateTime, default=get_current_time)

    accuracy = Column(Float)
    f1_score = Column(Float)
    model_file = Column(String)
    status = Column(String)
    config_id = Column(Integer, ForeignKey('data_split_configs.id', name='fk_model_config_id'))
    algorithm_id = Column(Integer, ForeignKey('algorithms.id', name='fk_model_algorithm_id'))

    config = relationship("DataSplitConfig", back_populates="models")
    algorithm = relationship("Algorithm", back_populates="models")