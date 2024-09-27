import os
from datetime import datetime

import pytz
from sqlalchemy import Column, DateTime, Float, Integer, String
from app.db.base_class import Base

UTC7 = pytz.timezone(os.getenv("SQLALCHEMY_DATABASE_CONNECT_TIMEZONE","Asia/Jakarta"))

def get_current_time():
    return datetime.now(UTC7)

class DataSplitConfig(Base):
    __tablename__ = 'data_split_configs'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    train_size = Column(Float)
    random_seed = Column(Integer)
    created_at = Column(DateTime, default=get_current_time)