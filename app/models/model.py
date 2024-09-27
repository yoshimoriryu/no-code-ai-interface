import os
from datetime import datetime

import pytz
from sqlalchemy import Column, DateTime, Integer, String
from app.db.base_class import Base
from sqlalchemy.sql.sqltypes import JSON

UTC7 = pytz.timezone(os.getenv("SQLALCHEMY_DATABASE_CONNECT_TIMEZONE","Asia/Jakarta"))

def get_current_time():
    return datetime.now(UTC7)

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    hyperparameters = Column(JSON)
    algorithm = Column(String)
    model_file = Column(String)
    created_at = Column(DateTime, default=get_current_time)
