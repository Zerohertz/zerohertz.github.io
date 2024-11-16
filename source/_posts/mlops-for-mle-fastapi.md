---
title: 'MLOps for MLE: FastAPI'
date: 2023-02-07 22:31:31
categories:
- 4. MLOps
tags:
- Python
- FastAPI
- Docker
---
# Introduction

~~~bash
$ conda activate MLOps
$ pip install "fastapi[all]"
~~~

~~~python Example.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
...
~~~

+ `app = FastAPI()`: `FastAPI` 클래스의 인스턴스 생성
+ Path Operation Decorator: API 작업의 endpoint를 HTTP method를 통해 지정
  + Operation: `POST`, `GET`, `PUT`, `DELETE` 등의 HTTP method
  + Ex. `@app.get("/")`: FastAPI가 path `/`에서 `GET` operation 수행
+ Path Operation Function: Path operation 수행 시 실행될 Python 함수
  + Return: `dict`, `list`, `str`, `int`, Pydantic Model, etc...

<!-- More -->

~~~
$ uvicorn Example:app --reload
~~~

+ `uvicorn`: FastAPI를 실행하는 웹 서버 실행 Command Line Tool
+ `Example`: `Example.py`
+ `app`: `Example.py`에서 선언된 `FastAPI`의 객체
+ `--reload`: 코드 변경 시 서버 재시작 옵션

<img width="762" alt="FastAPI: example_1" src="/images/mlops-for-mle-fastapi/218294927-b5e5265d-2176-4ff9-bc4d-57cbbce7dca5.png">

> http://localhost:8000

<img width="1141" alt="FastAPI: example_2" src="/images/mlops-for-mle-fastapi/217262283-76fa8515-5580-4c45-a67d-b7c5304d09f2.png">

> http://localhost:8000/doc

<img width="1141" alt="FastAPI: example_3" src="/images/mlops-for-mle-fastapi/217262336-51ced7dd-477e-4808-9549-45b730ededd5.png">

## Path Parameter

~~~python Example.py
...
@app.get("/items/{item_id}")
def read_item_parameter(item_id: int):
    return {"item_id": item_id}
...
~~~

> http://localhost:8000/items/202172279 & http://localhost:8000/items/hi

![Path Parameter](/images/mlops-for-mle-fastapi/217269004-676cdc8a-c1ec-43ce-af03-78d694f122cd.png)

## Query Parameter

~~~python Example.py
...
@app.get("/items/")
def read_item_query(skip: int = 0, limit: int = 10):
    test_db = [{"item_name": "ze", "item_name": "ro", "item_name": "hertz"}]
    return test_db[skip: skip + limit]
...
~~~

> http://localhost:8000/items/?skip=0&limit=10

<img width="1141" alt="Query Parameter" src="/images/mlops-for-mle-fastapi/217270657-2c58f378-5dd3-46c2-a8ad-f23b6e713fa3.png">

## Multiple Path and Query Parameters

~~~python
...
from typing import Union
...
@app.get("/users/{user_id}/items/{item_id}")
def read_user_item(user_id: int, item_id: str, q: Union[str, None] = None, short: bool = False):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"},
        )
    return item
~~~

> http://localhost:8000/users/3/items/zerohertz?q=hello&short=True & http://localhost:8000/users/3/items/zerohertz?short=True

![Multiple Path and Query Parameters 1](/images/mlops-for-mle-fastapi/217273288-deae06db-e96d-4b10-aa78-d473302200ed.png)

> http://localhost:8000/users/3/items/zerohertz?q=hello & http://localhost:8000/users/3/items/zerohertz

![Multiple Path and Query Parameters 2](/images/mlops-for-mle-fastapi/217275856-555c8b66-05f3-4aa7-b7c9-836b85336e8f.png)

---

# FastAPI CRUD

+ CRUD
  + C: Create
    + 이름과 별명을 입력하여 사용자 생성
  + R: Read
    + 이름을 입력하여 해당 이름을 가진 사용자의 별명 반환
    + 입력된 이름이 존재하지 않을 시 `400 status code` 및 `Name not found` 반환
    + `fastapi.HTTPException` 활용 가능
  + U: Update
    + 이름과 새로운 별명을 입력하여 해당 이름을 가진 사용자의 별명 업데이트
    + 입력된 이름이 존재하지 않을 시 `400 status code` 및 `Name not found` 반환
  + D: Delete
    + 이름을 입력하여 해당 이름을 가진 사용자 정보 삭제
+ Path Paramter: API에서 사용되는 파라미터를 `Request Header`에 입력
+ Query Parameter: API에서 사용되는 파라미터를 `Request Body`에 입력

## API 구현

~~~python CRUD_Path.py
from fastapi import FastAPI, HTTPException

app = FastAPI()
USER_DB = {}
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found.")

@app.post("/users/name/{name}/nickname/{nickname}")
def create_user(name: str, nickname: str):
    USER_DB[name] = nickname
    return {"status": "success"}

@app.get("/users/name/{name}")
def read_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    return {"nickname": USER_DB[name]}

@app.put("/users/name/{name}/nickname/{nickname}")
def update_user(name: str, nickname: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    USER_DB[name] = nickname
    return {"status": "success"}

@app.delete("/users/name/{name}")
def delete_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    del USER_DB[name]
    return {"status": "success"}
~~~

~~~python CRUD_Query.py
from fastapi import FastAPI, HTTPException

app = FastAPI()
USER_DB = {}
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found.")

@app.post("/users")
def create_user(name: str, nickname: str):
    USER_DB[name] = nickname
    return {"status": "success"}

@app.get("/users")
def read_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    return {"nickname": USER_DB[name]}

@app.put("/users")
def update_user(name: str, nickname: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    USER_DB[name] = nickname
    return {"status": "success"}

@app.delete("/users")
def delete_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    del USER_DB[name]
    return {"status": "success"}
~~~

![API Execution](/images/mlops-for-mle-fastapi/218295044-2f1e25fe-d197-427c-a229-6a033da48722.png)

## API 테스트

![API Test](/images/mlops-for-mle-fastapi/217852672-10ca0544-83f3-4c62-ba06-587d6f126f82.png)

---

# FastAPI CRUD (Pydantic)

+ Request Body: Client에서 API로 전송하는 데이터
+ Response Body: API에서 Client로 전송하는 데이터
+ Pydantic: Client와 API 사이에서 전송하는 데이터 형식을 지정하기 위한 모듈
  + 입력 받는 파라미터와 생성 후 반환하는 파라미터를 다르게 지정 가능
  + 비밀번호와 같은 사용자가 필수 입력해야 하지만 반환되면 안되는 파라미터 지정 시 사용

~~~python CRUD_Pydantic.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class CreateIn(BaseModel):
    name: str
    nickname: str

class CreateOut(BaseModel):
    status: str
    id: int

app = FastAPI()
USER_DB = {}
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found.")

@app.post("/users", response_model=CreateOut)
def create_user(user: CreateIn) -> CreateOut:
    USER_DB[user.name] = user.nickname
    user_dict = user.dict()
    user_dict["status"] = "success"
    user_dict["id"] = len(USER_DB)
    return user_dict

@app.get("/users")
def read_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    return {"nickname": USER_DB[name]}

@app.put("/users")
def update_user(name: str, nickname: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    USER_DB[name] = nickname
    return {"status": "success"}

@app.delete("/users")
def delete_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    del USER_DB[name]
    return {"status": "success"}
~~~

![FastAPI: CRUD (Pydantic)](/images/mlops-for-mle-fastapi/218295137-787b256e-9999-4173-be43-761192017f0c.png)

---

# FastAPI on Docker

~~~Docker CRUD_Pydantic.Dockerfile
FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip \
    && pip install "fastapi[all]"

COPY CRUD_Pydantic.py CRUD_Pydantic.py

CMD ["uvicorn", "CRUD_Pydantic:app", "--host", "0.0.0.0", "--reload"]
~~~

+ `RUN`: `pip` 업데이트 이후 `fastapi[all]` 설치
+ `COPY`: `CRUD_Pydantic.py`를 컨테이너 내부로 복사
+ `CMD`: `uvicorn`을 통해 `CRUD_Pydantic.py`의 FastAPI 객체 `app` 실행

~~~bash
$ docker build -t api-server -f CRUD_Pydantic.Dockerfile .
$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
api-server   latest    b80623bd7971   37 seconds ago   201MB
$ docker run -d \
--name api-server \
-p 8000:8000 \
api-server
$ docker ps
CONTAINER ID   IMAGE        COMMAND                  CREATED          STATUS          PORTS                    NAMES
f3c3a5dc60f7   api-server   "uvicorn CRUD_Pydant…"   18 seconds ago   Up 17 seconds   0.0.0.0:8000->8000/tcp   api-server
~~~

![FastAPI on Docker](/images/mlops-for-mle-fastapi/218295832-fa682e6e-05b6-425c-b856-772c842aed96.png)

---

# Reference

+ [MLOps for MLE: 05. FastAPI](https://mlops-for-mle.github.io/tutorial/docs/category/05-fastapi)