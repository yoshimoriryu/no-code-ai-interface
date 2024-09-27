import json
import os

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import SessionLocal, engine, get_db
from app.models import Model
from app.db.base_class import Base
from app.utils import clean_dict, load_existing_csv_files

# Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store CSV data with filename as the key
csv_storage = {}
load_existing_csv_files(csv_storage)

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse(content={"error": "File is not a CSV"}, status_code=400)

        file_location = f"uploaded_files/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        csv_storage[file.filename] = pd.read_csv(file_location)

        return {"filename": file.filename, "shape": csv_storage[file.filename].shape}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

    # Get the DataFrame
    df = csv_storage[filename]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Pagination logic
    total_rows = df.shape[0]

    # If pagination is disabled, reset page and page_size
    if not pagination_enabled:
        page = 1
        page_size = total_rows  # Fetch all records

    start = (page - 1) * page_size
    end = start + page_size

    # Ensure we don't go out of bounds
    if start >= total_rows:
        return JSONResponse(
            content={"error": "Page out of range"},
            status_code=400,
        )

    # Get the paginated or full data
    paginated_df = df.iloc[start:end] if pagination_enabled else df
    data_as_dict = paginated_df.to_dict(orient="records")
    cleaned_data = clean_dict(data_as_dict)

    # Check for missing values and missing columns
    missing_columns = paginated_df.columns[paginated_df.isnull().any() | (paginated_df == "").any()].tolist()
    has_missing_values = len(missing_columns) > 0

    num_instances = df.shape[0]  # Total instances
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


@app.get("/all_models/")
async def get_models(db: Session = Depends(get_db)):
    models = db.query(Model).all()
    return [
        {
            "id": model.id,
            "name": model.name,
            "algorithm": model.algorithm,
            "hyperparameters": model.hyperparameters or {},
        }
        for model in models
    ]

@app.get("/check-file/{filename}")
async def check_file(filename: str):
    # Check if the file exists in csv_storage
    exists = filename in csv_storage or f"{filename}.csv" in csv_storage
    return {"exists": exists}


@app.post("/models/")
def create_model(model: schemas.ModelCreate,  db: Session = Depends(get_db)):
    return crud.model.create_model(db=db, model=model)


@app.get("/models/{model_id}/")
def read_model(model_id: int,  db: Session = Depends(get_db)):
    return crud.model.get_model(db=db, model_id=model_id)


# @app.post("/train/")
# def train_model(model_id: int, data: schemas.TrainData):
#     return crud.train_model(db: Session = Depends(get_db), model_id=model_id, data=data)


# @app.post("/infer/")
# def infer(model_id: int, input_data: schemas.InferenceData):
#     return crud.infer_model(db: Session = Depends(get_db), model_id=model_id, input_data=input_data)

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
def list_data_splits(skip: int = 0, limit: int = 10,  db: Session = Depends(get_db)):
    return crud.data_split_config.get_all_data_split_configs(db=db, skip=skip, limit=limit)