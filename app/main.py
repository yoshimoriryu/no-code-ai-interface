import json
from app.logging_config import setup_logging
from typing import Annotated
import os
import pickle

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from saq

from app import crud, schemas
from app.database import SessionLocal, engine, get_db
from app.db.base_class import Base
from app.models import Model, Algorithm
from app.schemas.model import ModelUpdate
from app.utils.utils import clean_dict, load_existing_csv_files, extract_metadata_from_model

# Base.metadata.create_all(bind=engine)

app = FastAPI()

saq.configure(broker=os.getenv('SAQ_BROKER_URL', "redis://localhost:6379/0"))

origins_default = [
    "http://localhost:8080",
    "https://localhost:8080",
    ]
origins = os.getenv('BACKEND_CORS_ORIGINS', origins_default).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logging()

@app.middleware("https")
async def log_requests(request, call_next):
    response = await call_next(request)
    logger.info(f"Request: {request.method} {request.url} ======= {response.status_code}")
    return response

UPLOAD_FOLDER = "uploaded_files/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

csv_storage = {}
load_existing_csv_files(csv_storage)

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), upload_folder: str = ''):
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse(content={"error": "File is not a CSV"}, status_code=400)

        if upload_folder:
            file_location = f"{upload_folder}/{file.filename}"
        else:
            file_location = f"{UPLOAD_FOLDER}/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        csv_storage[file.filename] = pd.read_csv(file_location)

        return {"filename": file.filename, "shape": csv_storage[file.filename].shape}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/csv-metadata/")
async def get_csv_metadata(filename: str):
    file_location = f"{UPLOAD_FOLDER}{filename}"

    logger.info(file_location)

    df = pd.read_csv(file_location)
    total_rows = int(len(df))
    total_missing_values = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    feature_data_types = df.dtypes.reset_index()
    feature_data_types.columns = ['Feature', 'Data Type']
    feature_data_types['Data Type'] = feature_data_types['Data Type'].apply(lambda x: str(x)) 
    logger.info(feature_data_types)
    feature_data_types_dict = feature_data_types.to_dict(orient="records")
    
    return {
        'total_rows': total_rows,
        'total_missing_values': total_missing_values,
        'total_duplicates': total_duplicates,
        'feature_data_types': feature_data_types_dict
    }

@app.delete("/delete-file/")
async def delete_file(filename: str, db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(file_path):
        resp = crud.data_split_config.delete_config_by_filename(db=db, filename=filename)

        os.remove(file_path)
        removed_file = csv_storage.pop(filename, None)
        if removed_file is not None:
            return JSONResponse(content={"message": f"{filename} has been deleted"}, status_code=200)
        else:
            return HTTPException(status_code=404, detail="File not found")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/show-csv/{filename}")
async def show_csv(filename: str, page: int = 1, page_size: int = 10, pagination_enabled: bool = True):
    if filename not in csv_storage:
        if f"{filename}.csv" in csv_storage:
            filename = f"{filename}.csv"
        else:
            return JSONResponse(
                content={"error": "No CSV data uploaded with this filename"},
                status_code=404,
            )

    df = csv_storage[filename]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    total_rows = df.shape[0]

    if not pagination_enabled:
        page = 1
        page_size = total_rows

    start = (page - 1) * page_size
    end = start + page_size

    if start >= total_rows:
        return JSONResponse(
            content={"error": "Page out of range"},
            status_code=400,
        )

    paginated_df = df.iloc[start:end] if pagination_enabled else df
    data_as_dict = paginated_df.to_dict(orient="records")
    cleaned_data = clean_dict(data_as_dict)

    missing_columns = paginated_df.columns[paginated_df.isnull().any() | (paginated_df == "").any()].tolist()
    has_missing_values = len(missing_columns) > 0

    num_instances = df.shape[0]
    num_features = df.shape[1]

    response_data = {
        "filename": filename,
        "num_instances": num_instances,
        "num_features": num_features,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_rows + page_size - 1) // page_size if pagination_enabled else 1,  # Ceiling division
        "has_missing_values": has_missing_values,
        "missing_columns": missing_columns,
        "data": cleaned_data,
    }

    try:
        return JSONResponse(content=response_data)
    except ValueError as e:
        return JSONResponse(
            content={"error": f"Error serializing data to JSON: {str(e)}"},
            status_code=500,
        )

@app.get("/list-csv/")
async def list_csv():
    return {"uploaded_files": list(csv_storage.keys())}

@app.get("/get-csv-columns")
async def get_csv_columns(filename: str):
    if filename in csv_storage:
        df = csv_storage[filename]
        columns = df.columns.tolist()
        return {"columns": columns}
    else:
        return {"error": "File not found"}, 404

@app.get("/all-models/")
async def get_models(db: Session = Depends(get_db)):
    models = db.query(Model).all()

    retval = [
        {
        "id": model.id,
        "name": model.name,
        "hyperparameters": model.hyperparameters or {},
        "accuracy": model.accuracy,
        "f1_score": model.f1_score,
        "model_file": model.model_file,
        "status": model.status,
        "algorithm": model.algorithm,
        "config": model.config,
        "project_name": model.project_name,
        "created_at": model.created_at
    }
    for model in models
    ]

    for new_model, model in zip(retval, models):
        if model.config:
            new_model["config"] = {
                "filename": model.config.filename,
                "train_size": model.config.train_size,
                "random_seed": model.config.random_seed,
                "features": model.config.features,
                "target": model.config.target,
                "created_at": model.config.created_at
            }
        else:
            new_model["config"] = {
                "filename": None
            }

        if model.algorithm:
            new_model["algorithm"] = {
                "name": model.algorithm.name,
                "description": model.algorithm.description,
                "default_hyperparameters": model.algorithm.default_hyperparameters
            }

    return retval

@app.get("/check-file/{filename}")
async def check_file(filename: str):
    exists = filename in csv_storage or f"{filename}.csv" in csv_storage
    return {"exists": exists}


@app.post("/create-model/")
def create_model(model: schemas.ModelCreate,  db: Session = Depends(get_db)):
    logger.info('model config_id')
    logger.info(model.config_id)
    return crud.model.create_model(db=db, model=model)

@app.put("/update-model/{model_id}", response_model=schemas.ModelUpdate)
def update_model(
    model_id: int,
    model_update: schemas.ModelUpdate,
    db: Session = Depends(get_db)
):
    logger.info(f"Updating model with id: {model_id}")
    logger.info(f"Update data: {model_update}")

    existing_model = crud.model.get_model(db=db, model_id=model_id)
    if existing_model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    updated_model = crud.model.update_model(
        db=db,
        model_id=model_id,
        model_update=model_update
    )

    if updated_model is None:
        raise HTTPException(status_code=500, detail="Failed to update model")

    if model_update.model_file and model_update.model_file != existing_model.model_file:
        try:
            crud.train_model.delete_model_file(existing_model.model_file)
            crud.train_model.save_model_file(model_update.model_file)
        except Exception as e:
            logger.error(f"Error handling model file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return updated_model

@app.delete("/delete-model/{model_id}", response_model=schemas.Model)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    logger.info(db)
    logger.info(model_id)
    deleted_model = crud.model.delete_model(db=db, model_id=model_id)
    if deleted_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    # try:
    #     crud.train_model.delete_model_file(deleted_model.model_file)
    # except Exception as e:
    #     logger.info(e)
    #     raise HTTPException(status_code=404, detail=str(e))
    crud.train_model.delete_model_file(deleted_model.model_file)
    return deleted_model

@app.get("/models/{model_id}/")
def read_model(model_id: int,  db: Session = Depends(get_db)):
    return crud.model.get_model(db=db, model_id=model_id)

@app.post("/data-split-configs/", response_model=schemas.DataSplitConfig)
def create_data_split(config: schemas.DataSplitConfigCreate, db: Session = Depends(get_db)):
    return crud.data_split_config.create_data_split_config(db=db, config=config)

@app.get("/data-split-configs/{config_id}", response_model=schemas.DataSplitConfig)
def get_data_split(config_id: int,  db: Session = Depends(get_db)):
    db_config = crud.data_split_config.get_data_split_config(db=db, config_id=config_id)
    if db_config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return db_config

@app.get("/data-split-configs/", response_model=list[schemas.DataSplitConfig])
def list_data_splits(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.data_split_config.get_all_data_split_configs(db=db, skip=skip, limit=limit)

@app.delete("/delete-config/{config_id}", response_model=schemas.DataSplitConfig)
def delete_config(config_id: int, db: Session = Depends(get_db)):
    logger.info(db)
    logger.info(config_id)
    deleted_config = crud.data_split_config.delete_config_by_id(db=db, config_id=config_id)
    if deleted_config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return deleted_config

@app.post("/train-model/")
def train_model_api(
        model_id: int = Query(..., description="ID of the model to train"),
        config_id: int = Query(..., description="ID of the training configuration"),
        db: Session = Depends(get_db)
    ):
    try:
        logger.info('model train 0')
        result = crud.train_model.start_training(db, model_id, config_id)

        logger.info('model train 1')
        model_update = ModelUpdate(
            config_id=config_id,
            accuracy=result['training_result']['accuracy'],
            f1_score=result['training_result']['f1_score'],
            model_file=result['training_result']['model_file'],
            status=result['training_result']['status'],
        )

        logger.info('model train 2')
        return crud.model.update_model(db=db, model_id=model_id, model_update=model_update)
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/all-algorithms/")
async def get_algorithms(db: Session = Depends(get_db)):
    algorithms = db.query(Algorithm).all()

    algorithm_data = [
    {
        "id": algorithm.id,
        "name": algorithm.name,
        "algorithm": algorithm.name,
        "description": algorithm.description,
        "hyperparameters": algorithm.default_hyperparameters or {}
    }
    for algorithm in algorithms
    ]

    # Print only the first element
    logger.info(algorithm_data[0] if algorithm_data else "No algorithms found")
    return [
        {
            "id": algorithm.id,
            "name": algorithm.name,
            "algorithm": algorithm.name,
            "description": algorithm.description,
            "hyperparameters": algorithm.default_hyperparameters or {}
        }
        for algorithm in algorithms
    ]

@app.get("/download-model/{model_id}")
async def download_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == model_id).first()
    
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    model_file_path = (f"trained_models/{model.model_file}")
    
    return FileResponse(model_file_path, media_type='application/octet-stream', filename=model.model_file)

# TODO save uploaded model and store the model file
@app.post("/upload-model/")
async def upload_model(
    model_file: UploadFile = File(...),
):
    try:
        model_data = await model_file.read()
        model = pickle.loads(model_data)
        if hasattr(model, 'get_params'):
            algorithm = type(model).__name__
            hyperparameters = model.get_params()
        else:
            return JSONResponse({"error": "Unable to extract model metadata"}, status_code=400)

        return {
            "algorithm": algorithm,
            "hyperparameters": hyperparameters
        }
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/extract-model-details/")
async def extract_model_details(model_file: UploadFile = File(...)):
    try:
        contents = await model_file.read()
        model = pickle.loads(contents)
        algorithm = model.__class__.__name__ if hasattr(model, '__class__') else "Unknown Algorithm"
        hyperparameters = model.get_params() if hasattr(model, 'get_params') else {}

        return {
            "algorithm": algorithm,
            "hyperparameters": hyperparameters
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/inference-model/")
async def inference_model_api(
        model_id: int = Query(..., description="ID of the model to perform inference on"),
        config_id: int = Query(...),
        db: Session = Depends(get_db)
    ):
    try:
        logger.info('Starting inference for model ID: %d and config ID: %d', model_id, config_id)
        # TODO preprocess data file
        # currently using config to upload tobe infered files
        results = crud.infer_model(db=db, model_id=model_id, config_id=config_id)

        return results
    except Exception as e:
        logger.error('Error during inference: %s', e)
        raise HTTPException(status_code=400, detail=str(e))

