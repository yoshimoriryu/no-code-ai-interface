from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models import DataSplitConfig
from app.schemas import DataSplitConfigCreate, DataSplitConfigUpdate

class CRUDDataSplitConfig(
    CRUDBase[DataSplitConfig, DataSplitConfigCreate, DataSplitConfigUpdate]
):
  def create_data_split_config(self, db: Session, config: DataSplitConfigCreate):
      db_config = DataSplitConfig(
          filename=config.filename,
          train_size=config.train_size,
          random_seed=config.random_seed
      )
      db.add(db_config)
      db.commit()
      db.refresh(db_config)
      return db_config

  def get_data_split_config(self, db: Session, config_id: int):
      return db.query(DataSplitConfig).filter(DataSplitConfig.id == config_id).first()

  def get_all_data_split_configs(self, db: Session, skip: int = 0, limit: int = 10):
      return db.query(DataSplitConfig).offset(skip).limit(limit).all()

data_split_config = CRUDDataSplitConfig(DataSplitConfig)