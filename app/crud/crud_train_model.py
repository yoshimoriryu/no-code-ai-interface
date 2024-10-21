from sqlalchemy.orm import Session
import os
import logging
import stat

from app import crud
from app.utils.train_utils import perform_training


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

class CRUDTrainModel:
    def start_training(self, db: Session, model_id: int, config_id: int):
        model = crud.model.get_model(db, model_id)
        config = crud.data_split_config.get_data_split_config(db, config_id)

        if not model:
            raise ValueError(f"Model with ID {model_id} not found.")
        if not config:
            raise ValueError(f"Data Config with ID {config_id} not found.")

        training_result = perform_training(model, config)

        return {
            "message": "Training started",
            "model": model.name,
            "config": config.filename,
            "status": "in progress",
            "training_result": training_result
        }

    def log_file_permissions(self, file_path):
        permissions = stat.filemode(os.stat(file_path).st_mode)
        logger.info(f'Permissions for {file_path}: {permissions}')

    def delete_model_file(self, filename: str):
        model_path = os.path.abspath(f"trained_models/{filename}")
        self.log_file_permissions(model_path)  # Log permissions before deleting
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f'Successfully deleted: {model_path}')
            return {"message": "File deleted successfully", "status": 200}
        else:
            logger.warning(f'File not found: {model_path}')
            return {"message": "File not found", "status": 404}

train_model = CRUDTrainModel()