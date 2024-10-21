from sqlalchemy.exc import NoResultFound
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
            random_seed=config.random_seed,
            features=config.features,
            target=config.target,
            missing_data_strategy=config.missing_data_strategy,
            constant_value=config.constant_value
        )
        db.add(db_config)
        db.commit()
        db.refresh(db_config)
        return db_config

    def get_data_split_config(self, db: Session, config_id: int):
        return db.query(DataSplitConfig).filter(DataSplitConfig.id == config_id).first()

    def get_all_data_split_configs(self, db: Session, skip: int = 0, limit: int = 10):
        return db.query(DataSplitConfig).offset(skip).limit(limit).all()
  
    def delete_config_by_filename(self, db: Session, filename: str):
        try:
            # Delete all configurations with the specified filename
            result = db.query(DataSplitConfig).filter(DataSplitConfig.filename == filename).delete(synchronize_session=False)
            
            if result == 0:
                return {"message": "No configurations found with the specified filename."}

            db.commit()  # Commit the transaction to save changes

            return {"message": f"{result} configurations for '{filename}' have been deleted."}

        except Exception as e:
            db.rollback()  # Rollback in case of error
            return {"error": str(e)}

data_split_config = CRUDDataSplitConfig(DataSplitConfig)