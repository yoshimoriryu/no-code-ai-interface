from sqlalchemy.orm import Session
import logging
from app import crud
from app.utils.infer_utils import perform_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

class CRUDInferModel:
    def start_inference(self, db: Session, model_id: int, config_id: int):
        model = crud.model.get_model(db, model_id)

        if not model:
            raise ValueError(f"Model with ID {model_id} not found.")

        logger.info('start_inference: perform inference here')
        config = crud.data_split_config.get_data_split_config(db, config_id)
        prediction = perform_inference(model, config)

        return {
            "message": "Inference completed",
            "model": model.name,
            "status": "completed",
            "prediction": prediction
        }

infer_model = CRUDInferModel()
