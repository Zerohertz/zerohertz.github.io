---
title: 'MLOps for MLE: API Serving'
date: 2023-02-15 20:51:19
categories:
- 4. MLOps
tags:
- Python
- MLflow
- FastAPI
- Docker
---
# Model API

## 모델 다운로드

~~~python download_model.py
import os
from dotenv import load_dotenv
from argparse import ArgumentParser

import mlflow

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI")
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD")

def download_model(args):
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{args.run_id}/{args.model_name}", dst_path=".")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
    parser.add_argument("--run-id", dest="run_id", type=str)
    args = parser.parse_args()
    download_model(args)
~~~

<!-- More -->

+ `mlflow.artifacts.download_artifacts(artifact_uri, run_id, artifact_path, dst_path, tracking_uri)`: Download an artifact file or directory to a local directory
  + `artifact_uri`: URI pointing to the artifacts
  + `run_id`: ID of the MLflow Run containing the artifacts
  + `artifact_path`: (For use with `run_id`) If specified, a path relative to the MLflow 
  + `dst_path`: Path of the local filesystem destination directory to which download the specified artifacts
  + `tracking_uri`: The tracking URI to be used when downloading artifacts
  + Model artifact: MLflow에 모델이 저장될 때 함께 저장된 메타데이터와 모델 자체의 binary 파일

~~~bash
/MLOps-Breast-Cancer/Database$ docker compose up -d
/MLOps-Breast-Cancer/Database$ docker ps
CONTAINER ID   IMAGE                     COMMAND                  CREATED              STATUS                        PORTS                              NAMES
3402992a232b   database-data-generator   "python data_generat…"   About a minute ago   Up About a minute                                                data-generator
c8acce9e6021   postgres:14.0             "docker-entrypoint.s…"   About a minute ago   Up About a minute (healthy)   0.0.0.0:1234->5432/tcp             postgres-server
/MLOps-Breast-Cancer/MLflow$ docker compose up -d
/MLOps-Breast-Cancer/MLflow$ docker ps
CONTAINER ID   IMAGE                  COMMAND                  CREATED          STATUS                    PORTS                              NAMES
e44a64bbd401   mlflow-mlflow-server   "/bin/sh -c 'mc conf…"   43 seconds ago   Up 11 seconds             0.0.0.0:5001->5000/tcp             mlflow-server
23805103885c   minio/minio            "/usr/bin/docker-ent…"   43 seconds ago   Up 42 seconds (healthy)   0.0.0.0:9000-9001->9000-9001/tcp   mlflow-artifact-store
e3a31b3fa8ee   postgres:14.0          "docker-entrypoint.s…"   43 seconds ago   Up 42 seconds (healthy)   5432/tcp                           mlflow-backend-store
/MLOps-Breast-Cancer/MLflow$ python save_model_to_registry.py --model-name "sk_model"
save_model_to_registry.py:28: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql("SELECT * FROM breast_cancer_data ORDER BY id DESC LIMIT 100", db_connect)
Train Accuracy : 0.9875
Valid Accuracy : 0.95
2023/02/15 21:23:37 INFO mlflow.tracking.fluent: Experiment with name 'new-exp' does not exist. Creating a new experiment.
/Users/zerohertz/miniforge3/envs/MLOps/lib/python3.8/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
~~~

![save-model-to-registry](/images/mlops-for-mle-api-serving/save-model-to-registry.png)

~~~bash
/MLOps-Breast-Cancer/MLflow$ python download_model.py --model-name sk_model --run-id a8ec09bf9c4b4b5ab82ddde01a14109b
/MLOps-Breast-Cancer/MLflow/sk_model$ ls
MLmodel            conda.yaml         input_example.json model.pkl          python_env.yaml    requirements.txt
~~~

## API 구현

~~~python schemas.py
from pydantic import BaseModel

class PredictIn(BaseModel):
    feature_a: float
    feature_b: float
    feature_c: float
    feature_d: float
    feature_e: float
    feature_f: float
    feature_g: float
    feature_h: float
    feature_i: float
    feature_j: float
    feature_k: float
    feature_l: float
    feature_m: float
    feature_n: float
    feature_o: float
    feature_p: float
    feature_q: float
    feature_r: float
    feature_s: float
    feature_t: float
    feature_u: float
    feature_v: float
    feature_w: float
    feature_x: float
    feature_y: float
    feature_z: float
    feature_aa: float
    feature_bb: float
    feature_cc: float
    feature_dd: float

class PredictOut(BaseModel):
    target: int
~~~

~~~python app.py
import mlflow
import pandas as pd
from fastapi import FastAPI
from schemas import PredictIn, PredictOut

def get_model():
    model = mlflow.sklearn.load_model(model_uri="../MLflow/sk_model")
    return model

MODEL = get_model()
app = FastAPI()

@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    pred = MODEL.predict(df).item()
    return PredictOut(target=pred)
~~~

## API Execution

~~~bash
/MLOps-Breast-Cancer/FastAPI$ uvicorn app:app --reload
~~~

![results](/images/mlops-for-mle-api-serving/results.png)

---

# Model API on Docker Compose

~~~Docker FastAPI.Dockerfile
FROM amd64/python:3.8-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install mlflow==1.30.0 pandas scikit-learn "fastapi[all]"

COPY schemas.py schemas.py
COPY app.py app.py
COPY sk_model/ sk_model/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload"]
~~~

~~~yaml docker-compose.yaml
version: "3"

services:
  api-with-model:
    build:
      context: .
      dockerfile: FastAPI.Dockerfile
    container_name: api-with-model
    ports:
      - 8000:8000
    healthcheck:
      test:
        - curl -X POST http://localhost:8000/predict
        - -H
        - "Content-Type: application/json"
        - -d
        - '{"feature_a": 12.77, "feature_b": 29.43, "feature_c": 81.35, "feature_d": 507.9, "feature_e": 0.08276, "feature_f": 0.04234, "feature_g": 0.01997, "feature_h": 0.01499, "feature_i": 0.1539, "feature_j": 0.05637, "feature_k": 0.2409, "feature_l": 1.367, "feature_m": 1.477, "feature_n": 18.76, "feature_o": 0.008835, "feature_p": 0.01233, "feature_q": 0.01328, "feature_r": 0.009305, "feature_s": 0.01897, "feature_t": 0.001726, "feature_u": 13.87, "feature_v": 36.0, "feature_w": 88.1, "feature_x": 594.7, "feature_y": 0.1234, "feature_z": 0.1064, "feature_aa": 0.08653, "feature_bb": 0.06498, "feature_cc": 0.2407, "feature_dd": 0.06484}'
      interval: 10s
      timeout: 5s
      retries: 5

networks:
  default:
    name: mlops-network
    external: true
~~~

~~~bash
/MLOps-Breast-Cancer/FastAPI$ docker compose up -d
/MLOps-Breast-Cancer/FastAPI$ docker ps
CONTAINER ID   IMAGE                    COMMAND                  CREATED             STATUS                       PORTS                              NAMES
acd6af155bb2   fastapi-api-with-model   "uvicorn app:app --h…"   8 seconds ago       Up 7 seconds                 0.0.0.0:8000->8000/tcp             api-with-model

/MLOps-Breast-Cancer$ curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature_a": 22.27, "feature_b": 19.67, "feature_c": 152.8, "feature_d": 1509.0, "feature_e": 0.1326, "feature_f": 0.2768, "feature_g": 0.4264, "feature_h": 0.1823, "feature_i": 0.2556, "feature_j": 0.07039, "feature_k": 1.215, "feature_l": 1.545, "feature_m": 10.05, "feature_n": 170.0, "feature_o": 0.006515, "feature_p": 0.08668, "feature_q": 0.104, "feature_r": 0.0248, "feature_s": 0.03112, "feature_t": 0.005037, "feature_u": 28.4, "feature_v": 28.01, "feature_w": 206.8, "feature_x": 2360.0, "feature_y": 0.1701, "feature_z": 0.6997, "feature_aa": 0.9608, "feature_bb": 0.291, "feature_cc": 0.4055, "feature_dd": 0.09789}'
{"target":0}%

/MLOps-Breast-Cancer$ curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature_a": 11.14, "feature_b": 14.07, "feature_c": 71.24, "feature_d": 384.6, "feature_e": 0.07274, "feature_f": 0.06064, "feature_g": 0.04505, "feature_h": 0.01471, "feature_i": 0.169, "feature_j": 0.06083, "feature_k": 0.4222, "feature_l": 0.8092, "feature_m": 3.33, "feature_n": 28.84, "feature_o": 0.005541, "feature_p": 0.03387, "feature_q": 0.04505, "feature_r": 0.01471, "feature_s": 0.03102, "feature_t": 0.004831, "feature_u": 12.12, "feature_v": 15.82, "feature_w": 79.62, "feature_x": 453.5, "feature_y": 0.08864, "feature_z": 0.1256, "feature_aa": 0.1201, "feature_bb": 0.03922, "feature_cc": 0.2576, "feature_dd": 0.07018}'
{"target":1}%
~~~

---

# Reference

+ [MLOps for MLE: 06. API Serving](https://mlops-for-mle.github.io/tutorial/docs/category/06-api-serving)