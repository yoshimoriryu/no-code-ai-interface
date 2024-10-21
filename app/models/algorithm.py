import os
from datetime import datetime

import pytz
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import JSON

from app.db.base_class import Base

UTC7 = pytz.timezone(os.getenv("SQLALCHEMY_DATABASE_CONNECT_TIMEZONE","Asia/Jakarta"))

def get_current_time():
    return datetime.now(UTC7)


class Algorithm(Base):
    __tablename__ = "algorithms"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True, index=True)  # e.g., "Random Forest", "SVM"
    description = Column(String, nullable=True)
    default_hyperparameters = Column(JSON)  # Default hyperparameters for this algorithm (tied to this algorithm)
    created_at = Column(DateTime, default=get_current_time)

    models = relationship("Model", back_populates="algorithm")