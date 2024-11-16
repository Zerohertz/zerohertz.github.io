---
title: 'MLOps for MLE: Model Registry'
date: 2023-02-04 15:19:33
categories:
- 4. MLOps
tags:
- Python
- Docker
- MLflow
---
# MLflow Setup

+ [MLflow](https://github.com/mlflow/mlflow/): A machine learning lifecycle platform
  + MLflow Tracking: 학습 과정에서의 매개 변수, 코드, 결과를 기록하고 대화형 UI로 비교할 수 있는 API
  + MLflow project: ML 코드 공유를 위한 Conda 및 Docker 기반 패키지 형식 지원
  + MLflow Models: Docker, Apache Spark, Azure ML, AWS SageMaker 등의 플랫폼에 ML 코드를 배포할 수 있는 패키징 형식 및 도구
  + MLflow Model Registry: MLflow 모델의 전체 수명 주기를 공동으로 관리하기 위한 중앙 집중식 모델 저장소, API, UI의 집합
+ Backend Store: 수치 데이터와 MLflow 서버의 정보들을 체계적으로 관리하기 위한 DB
  + 모델 데이터
    + 학습 결과
      + Accuracy
      + F1-score
    + 학습 과정의 loss
    + 모델 자체 정보 (hyperparameters)
  + MLflow의 메타 데이터
    + run_id
    + run_name
    + experiment_name

<!-- More -->

## PostgreSQL DB Server

~~~yaml docker-compose.yaml
...
services:
  mlflow-backend-store:
    image: postgres:14.0
    container_name: mlflow-backend-store
    environment:
      POSTGRES_USER: zerohertz
      POSTGRES_PASSWORD: qwer123
      POSTGRES_DB: mlflowdatabase
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "zerohertz", "-d", "mlflowdatabase"]
      interval: 10s
      timeout: 5s
      retries: 5
...
~~~

+ `image`: `postgres:14.0`
+ `environment`
  + `POSTGRES_USER`: DB 접근을 위한 사용자 이름
  + `POSTGRES_PASSWORD`: DB 접근을 위한 비밀번호
  + `POSTGRES_DB`: DB 이름 설정
+ `healthcheck`
  + DB 서버 상태 체크

## MLflow Artifact Store

+ Artifact Store: MLFlow에서 학습된 모델을 저장하는 Model Registry로 이용하기 위한 스토리지 서버
  + 기본 파일 시스템에 비해 체계적 관리 가능
  + 외부의 스토리지 서버 사용 가능
+ MinIO: AWS S3를 대체할 수 있는 오픈 소스 고성는 개체 스토리지
  + AWS S3의 API와 호환되어 SDK 동일 사용 가능
  + MLflow에서는 AWS S3를 모델 저장 스토리지 사용 권장

~~~yaml docker-compose.yaml
...
services:
...
  mlflow-artifact-store:
    image: minio/minio
    container_name: mlflow-artifact-store
    ports:
      - 9000:9000
      - 9001:9001
    environment:
      MINIO_ROOT_USER: zerohertz_minio
      MINIO_ROOT_PASSWORD: asdf456!
    command: server /data/minio --console-address :9001
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
...
~~~

+ `image`: `minio/minio`
+ `ports`
  + `9000`: MinIO의 API 포트
  + `9001`: Console 포트
+ `environment`
  + `MINIO_ROOT_USER`: MinIO 접근을 위한 사용자 이름
  + `MINIO_ROOT_PASSWORD`: MinIO 접근을 위한 비밀번호
+ `command`
  + MinIO 서버 실행 명령어 추가
  + `--console-address`: `9001` 포트로 MinIO에 접근
+ `healthcheck`
  + MinIO 서버 상태 체크

## MLflow Server

~~~Docker MLflow.Dockerfile
FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip &&\
    pip install boto3==1.26.8 mlflow==1.30.0 psycopg2-binary

RUN cd /tmp && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/bin/mc
~~~

+ `FROM`
  + Python 3.9가 포함된 이미지를 BASE 이미지로 설정
+ `RUN`
  + `git`: MLflow 서버 내부 동작
  + `wget`: MinIO Client 설치
+ `RUN`
  + MLflow, PostgreSQL DB, AWS S3 관련 Python 패키지 설치
+ `RUN`
  + `wget`을 통해 MinIO Client 설치

~~~yaml docker-compose.yaml
...
services:
...
  mlflow-server:
    build:
      context: .
      dockerfile: MLflow.Dockerfile
    container_name: mlflow-server
    depends_on:
      mlflow-backend-store:
        condition: service_healthy
      mlflow-artifact-store:
        condition: service_healthy
    ports:
      - 5001:5000
    environment:
      AWS_ACCESS_KEY_ID: zerohertz_minio
      AWS_SECRET_ACCESS_KEY: asdf456!
      MLFLOW_S3_ENDPOINT_URL: http://mlflow-artifact-store:9000
    command:
      - /bin/sh
      - -c
      - |
        mc config host add mlflowminio http://mlflow-artifact-store:9000 zerohertz_minio asdf456! &&
        mc mb --ignore-existing mlflowminio/mlflow
        mlflow server \
        --backend-store-uri postgresql://zerohertz:qwer123@mlflow-backend-store/mlflowdatabase \
        --default-artifact-root s3://mlflow/ \
        --host 0.0.0.0
~~~

+ `depends_on`
  + `mlflow-server`의 서버 실행 전 `mlflow-backend-store`, `mlflow-artifact-store`을 먼저 실행
+ `ports`
  + `5001:5000` 포트 설정
+ `environment`
  + `AWS_ACCESS_KEY_ID`: AWS S3의 credential 정보 (= `MINIO_ROOT_USER`)
  + `AWS_SECRET_ACCESS_KEY`: AWS S3의 credentail 정보 (= `MINIO_ROOT_PASSWORD`)
  + `MLFLOW_S3_ENDPOINT_URL`: AWS S3의 주소 설정 (= MinIO의 주소)
    + `https://`로 정의해주면 오류 발생,,,
+ `command`: MinIO 초기 버켓 생성, MLflow 서버 실행
  + `mc config ~`: MinIO Client 기반 MinIO 서버에 호스트 등록
  + `mc mb ~`: 등록된 호스트를 통해 초기 버켓 생성
  + `mlflow server`: MLflow 서버 실행
  + `--backend-store-uri`: PostgreSQL DB와 연결
    + URI에 특수문자 기용 시 오류 발생 주의
  + `--default-artifact-root`: 명시된 버켓을 통해 MinIO의 초기 버켓과 연결

## Execution

![MinIO](/images/mlops-for-mle-model-registry/216772172-58442c28-244e-49a7-b0d3-4fe1c3239fed.png)


![MLflow](/images/mlops-for-mle-model-registry/216772178-a27b52e3-2478-436c-a5de-e58bbea27276.png)

---

# Save Model to Registry

## 환경 변수 설정

~~~python save_model_to_registry.py
import os
...
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "zerohertz_minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "asdf456!"
...
~~~

+ `MLFLOW_S3_ENDPOINT_URL`: 모델을 저장할 스토리지의 주소 (MinIO API)
+ `MLFLOW_TRACKING_URI`: 정보를 저장하기 위해 연결할 MLflow 서버의 주소
+ `AWS_ACCESS_KEY_ID`: MinIO에 접근하기 위한 아이디
+ `AWS_SECRET_ACCESS_KEY`: MinIO에 접근하기 위한 비밀번호

## 모델 저장

~~~python save_model_to_registry.py
...
from argparse import ArgumentParser
...
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
args = parser.parse_args()

mlflow.set_experiment("new-exp")

signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train.iloc[:10]

with mlflow.start_run():
    mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path=args.model_name,
        signature=signature,
        input_example=input_sample,
    )
...
~~~

+ `mlflow.set_experiment(experiment_name)`: `experiment`가 존재하지 않은 경우 새로 생성, 존재하는 경우 해당 `experiment` 사용
+ `mlflow.models.signature.infer_signature(model_input, model_output)`: 잘못된 데이터 입력 시 에러 발생을 위한 정보 입력
+ `mlflow.start_run(run_id, experiment_id, run_name, nested, tags, description)`: 새로운 MLflow `run` 시작
+ `mlflow.log_metrics(metrics, step)`: 모델의 결과 metrics를 `dictionary` 형태로 입력해 생성된 `run`에 저장
+ `mlflow.sklearn.log_model(sk_model, artifact_path, conda_env, code_paths, serialization_format, registered_model_name, signature, input_example, ...)`: 현재 `run`에 대해 `sklearn` 모델 기록

## Result

~~~bash
$ python save_model_to_registry.py --model-name "sk_model"
save_model_to_registry.py:25: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql("SELECT * FROM breast_cancer_data ORDER BY id DESC LIMIT 100", db_connect)
Train Accuracy : 0.9875
Valid Accuracy : 0.95
2023/02/05 00:42:10 INFO mlflow.tracking.fluent: Experiment with name 'new-exp' does not exist. Creating a new experiment.
/Users/zerohertz/miniforge3/envs/MLOps/lib/python3.8/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
~~~

![MiniO: Saved Model](/images/mlops-for-mle-model-registry/216777098-b68fabe9-d195-4e4e-ae79-57fcc8b52acc.png)

![MLflow: Saved Model 1](/images/mlops-for-mle-model-registry/216776431-ba8f15a1-e04d-4c9d-84cf-3280908e4f61.png)

![MLflow: Saved Model 2](/images/mlops-for-mle-model-registry/216776459-3f396912-b07f-4062-bf56-d14efc9266c6.png)

---

# Load Model from Registry

## 환경 변수 설정

~~~python load_model_from_registry.py
import os
...
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "zerohertz_minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "asdf456!"
...
~~~

## 모델 불러오기

~~~python load_model_from_registry.py
...
from argparse import ArgumentParser
...
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
parser.add_argument("--run-id", dest="run_id", type=str)
args = parser.parse_args()

model_pipeline = mlflow.sklearn.load_model(f"runs:/{args.run_id}/{args.model_name}")
...
~~~

![MLflow: RUN ID](/images/mlops-for-mle-model-registry/216776808-3f8e962d-b8bd-4170-a4a6-54ec142ed8e0.png)

+ `mlflow.sklearn.load_model(model_uri, dst_path)`: 로컬 파일 또는 `run`에서 `sklearn` 모델 load
+ `mlflow.pyfunc.load_model(model_uri, suppress_warnings, dst_path)`: Python 함수 형태로 저장된 모델 load

## Result

~~~bash
$ python load_model_from_registry.py --model-name "sk_model" --run-id 12abbfc176364e198849473c83783070
Train Accuracy : 0.9875
Valid Accuracy : 0.95
~~~

---

# [.env를 통한 환경 변수 정의](https://github.com/Zerohertz/MLOps-Breast-Cancer/commit/edc59d6da673f04c42728693b28088cf3f2ce2c5)

~~~bash .env
POSTGRES_USER=zerohertz_postgres
POSTGRES_PASSWORD=qwer123!
POSTGRES_DB=mlflowdatabase
MINIO_ROOT_USER=zerohertz_minio
MINIO_ROOT_PASSWORD=asdf456!
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
MLFLOW_TRACKING_URI=http://localhost:5001
~~~

~~~yaml docker-compose.yaml
...
services:
  mlflow-backend-store:
...
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "${POSTGRES_USER}", "-d", "${POSTGRES_DB}"]
...
  mlflow-artifact-store:
...
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
...
  mlflow-server:
    environment:
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
...
    command:
...
        mc config host add mlflowminio http://mlflow-artifact-store:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} &&
...
        --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@mlflow-backend-store/${POSTGRES_DB} \
...
~~~

~~~python save_model_registry.py & load_model_from_registry.py
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI")
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD")
~~~

---

# Reference

+ [MLOps for MLE: 03. Model Registry](https://mlops-for-mle.github.io/tutorial/docs/category/03-model-registry)