---
title: 'MLOps for MLE: Database'
date: 2023-01-21 16:56:36
categories:
- 4. MLOps
tags:
- Python
- Docker
---
# DB Server Creation

## DB 서버 생성

~~~bash
$ docker run -d \
  --name postgres-server \
  -p 5432:5432 \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=mydatabase \
  postgres:14.0
~~~

+ `-d`: 컨테이너 detached 모드 실행
+ `--name`: 컨테이너의 이름 정의
+ `-p`: Port forwarding 설정 $\rightarrow$ `host:container`
+ `-e`: 환경 변수 설정
  + `POSTGRES_USER`: 유저 이름 설정
  + `POSTGRES_PASSWORD`: 유저 비밀번호 설정
  + `POSTGRES_DB`: DB의 이름 설정
+ `postgres:14.0`: 사용 이미지 지정

<!-- More -->

<img width="651" alt="docker run" src="/images/mlops-for-mle-database/213859104-97231fc0-f9d9-41a9-9852-01e8351b7f5e.png">

<img width="1382" alt="Result" src="/images/mlops-for-mle-database/213859147-7103a19f-fa35-4173-ab3f-7db1bbe1115c.png">

~~~bash
$ docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED         STATUS         PORTS                    NAMES
056290ef2b93   postgres:14.0   "docker-entrypoint.s…"   5 minutes ago   Up 5 minutes   0.0.0.0:5432->5432/tcp   postgres-server
~~~

+ `postgres-server`라는 이름을 가진 컨테이너가 잘 실행되고 있음을 확인 완료

> 컨테이너 재실행 시

~~~bash
$ docker start postgres-server
~~~

## DB 서버 확인

~~~bash
$ psql --version
zsh: command not found: psql
~~~

+ PostgreSQL DB 서버 확인 CLI 툴인 `psql`이 설치되어있지 않아 아래 절차 수행

~~~bash
$ brew install libpq
$ echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
$ source ~/.zshrc
$ psql --version
psql (PostgreSQL) 15.1
~~~

~~~bash
$ PGPASSWORD=mypassword psql -h localhost -p 5432 -U myuser -d mydatabase
psql (15.1, server 14.0 (Debian 14.0-1.pgdg110+1))
Type "help" for help.

mydatabase=#
~~~

+ `PGPASSWORD=`: 접속할 유저의 비밀번호 입력
+ `-h`: 호스트 지정
+ `-p`: 포트 지정
+ `-U`: 유저 이름 지정
+ `-d`: DB의 이름 입력

<img width="716" alt="\du" src="/images/mlops-for-mle-database/213860629-036f95f5-e112-48b1-a4e3-498d24082b51.png">

+ `\du` 명령어를 통해 컨테이너에 접속한 현재 상태 조회

---

# Table Creation

## Python 가상 환경 설정 및 패키지 설치

~~~bash
$ conda create -n MLOps python=3.8
$ conda activate MLOps
$ pip install pandas psycopg2-binary scikit-learn
~~~

## 테이블 생성

+ `psycopg2`: PostgreSQL DB 서버에 접속하는 python library

## DB Connection

~~~python
>>> import psycopg2
>>> db_connect = psycopg2.connect(
...     user="myuser",
...     password="mypassword",
...     host="localhost",
...     port=5432,
...     database="mydatabase",
... )
>>> db_connect
<connection object at 0x10184f6d0; dsn: 'user=myuser password=xxx dbname=mydatabase host=localhost port=5432', closed: 0>
~~~

## Table Creation Query

~~~python
>>> import pandas as pd
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y = True, as_frame = True)
>>> X
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8

[150 rows x 4 columns]
>>> y
0      0
1      0
2      0
3      0
4      0
      ..
145    2
146    2
147    2
148    2
149    2
Name: target, Length: 150, dtype: int64
>>> df = pd.concat([X, y], axis = 'columns')
>>> df
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                  5.1               3.5                1.4               0.2       0
1                  4.9               3.0                1.4               0.2       0
2                  4.7               3.2                1.3               0.2       0
3                  4.6               3.1                1.5               0.2       0
4                  5.0               3.6                1.4               0.2       0
..                 ...               ...                ...               ...     ...
145                6.7               3.0                5.2               2.3       2
146                6.3               2.5                5.0               1.9       2
147                6.5               3.0                5.2               2.0       2
148                6.2               3.4                5.4               2.3       2
149                5.9               3.0                5.1               1.8       2

[150 rows x 5 columns]
>>> print(df.dtypes)
sepal length (cm)    float64
sepal width (cm)     float64
petal length (cm)    float64
petal width (cm)     float64
target                 int64
dtype: object
>>> create_table_query = """
... CREATE TABLE IF NOT EXISTS iris_data (
...     id SERIAL PRIMARY KEY,
...     timestamp timestamp,
...     sepal_length float8,
...     sepal_width float8,
...     petal_length float8,
...     petal_width float8,
...     target int
... );"""
>>> create_table_query
'\nCREATE TABLE IF NOT EXISTS iris_data (\n    id SERIAL PRIMARY KEY,\n    timestamp timestamp,\n    sepal_length float8,\n    sepal_width float8,\n    petal_length float8,\n    petal_width float8,\n    target int\n);'
~~~

### Send Query to DB

~~~python
>>> cur = db_connect.cursor() # psycopg2.connect.cursor 인스턴스 생성
>>> cur
<cursor object at 0x162613310; closed: 0>
>>> cur.execute(create_table_query) # psycopg2.connect의 cursor를 통해 query 전달
>>> db_connect.commit() # Query 실행
>>> cur.close() # Cursor 사용 종료
~~~

~~~python
with db_connect.cursor() as cur:  
    cur.execute(create_table_query)  
    db_connect.commit()
~~~

위 두 코드는 동일 !

### Table Creator

~~~python
def create_table(db_connect):  
    create_table_query = """  
    CREATE TABLE IF NOT EXISTS iris_data (  
        id SERIAL PRIMARY KEY,  
        timestamp timestamp,  
        sepal_length float8,  
        sepal_width float8,  
        petal_length float8,  
        petal_width float8,  
        target int  
    );"""  
    print(create_table_query)  
    with db_connect.cursor() as cur:  
        cur.execute(create_table_query)  
        db_connect.commit()
~~~

현재까지의 프로세스를 한 함수에 축약하면 위와 같고, `db_connect`를 입력받아 DB와의 지속적 연결 시 부하를 최소화

## 테이블 확인

~~~sql
mydatabase=# \d
               List of relations
 Schema |       Name       |   Type   | Owner
--------+------------------+----------+--------
 public | iris_data        | table    | myuser
 public | iris_data_id_seq | sequence | myuser
(2 rows)

mydatabase=# select * from iris_data;
 id | timestamp | sepal_length | sepal_width | petal_length | petal_width | target
----+-----------+--------------+-------------+--------------+-------------+--------
(0 rows)
~~~

---

# Data Insertion

~~~python
>>> def insert_data(db_connect, data):
...     insert_row_query = f""" # 데이터 삽입을 위한 query문 작성
...     INSERT INTO iris_data
...         (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
...         VALUES (
...             NOW(),
...             {data.sepal_length},
...             {data.sepal_width},
...             {data.petal_length},
...             {data.petal_width},
...             {data.target}
...         );"""
...     print(insert_row_query)
...     with db_connect.cursor() as cur:
...         cur.execute(insert_row_query)
...         db_connect.commit()
>>> insert_data(db_connect, df.sample(1).squeeze())

    INSERT INTO iris_data
        (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (
            NOW(),
            5.0,
            2.3,
            3.3,
            1.0,
            1.0
        );
~~~

~~~sql
mydatabase=# select * from iris_data;
 id |         timestamp          | sepal_length | sepal_width | petal_length | petal_width | target
----+----------------------------+--------------+-------------+--------------+-------------+--------
  1 | 2023-01-22 15:41:22.084967 |            5 |         2.3 |          3.3 |           1 |      1
(1 row)
~~~

---

# Data Insertion Loop

~~~python
>>> import time
>>>
>>> def generate_data(db_connect, df):
...     while True:
...             insert_data(db_connect, df.sample(1).squeeze())
...             time.sleep(1)
>>> generate_data(db_connect, df)

    INSERT INTO iris_data
        (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (
            NOW(),
            6.5,
            2.8,
            4.6,
            1.5,
            1.0
        );

    INSERT INTO iris_data
        (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (
            NOW(),
            5.5,
            3.5,
            1.3,
            0.2,
            0.0
        );

    INSERT INTO iris_data
        (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (
            NOW(),
            6.1,
            2.6,
            5.6,
            1.4,
            2.0
        );
~~~

~~~sql
mydatabase=# select * from iris_data;
 id |         timestamp          | sepal_length | sepal_width | petal_length | petal_width | target
----+----------------------------+--------------+-------------+--------------+-------------+--------
  1 | 2023-01-22 15:41:22.084967 |            5 |         2.3 |          3.3 |           1 |      1
  2 | 2023-01-22 15:46:07.344343 |          6.5 |         2.8 |          4.6 |         1.5 |      1
  3 | 2023-01-22 15:46:08.362642 |          5.5 |         3.5 |          1.3 |         0.2 |      0
  4 | 2023-01-22 15:46:09.378704 |          6.1 |         2.6 |          5.6 |         1.4 |      2
  5 | 2023-01-22 15:46:10.40752  |          5.2 |         2.7 |          3.9 |         1.4 |      1
...
 30 | 2023-01-22 15:46:35.993371 |          6.4 |         2.9 |          4.3 |         1.3 |      1
 31 | 2023-01-22 15:46:37.015506 |          4.9 |         2.5 |          4.5 |         1.7 |      2
 32 | 2023-01-22 15:46:38.073922 |          5.1 |         3.7 |          1.5 |         0.4 |      0
 33 | 2023-01-22 15:46:39.091883 |          5.6 |         2.8 |          4.9 |           2 |      2
 34 | 2023-01-22 15:46:40.105631 |          4.8 |           3 |          1.4 |         0.1 |      0
(34 rows)
~~~

---

# Data Generator on Docker

+ 두 컨테이너 (`DB Container`, `Data Generator`) 사이에서 통신이 불가한 이유
  + `DB Container`의 `5432` 포트에서 `localhost:5432` 포트로 통신
  + `Data Generator`의 `5432` 포트에서 `localhost:5432` 포트로 통신
+ `docker network`
  + 통신을 위한 컨테이너 이름 전달 필수
  + 컨테이너가 종료된 경우 사용된 이름을 초기화하기 위해 종료된 컨테이너 삭제 후 재실행 필요
  + 컨테이너 실행 순서 보장 불가 (`DB Container`가 먼저 실행되지 않는 경우 오류 발생)

Container Orchestration (여러 컨테이너 작업 조율)을 위해 `Docker Compose` 사용 !

---

# Data Generator on Docker Compose

~~~yaml docker-compose.yaml
version: "3"

services:
  postgres-server:
    image: postgres:14.0
    container_name: postgres-server
    ports:
      - 1234:5432
    environment:
      POSTGRES_USER: zerohertz
      POSTGRES_PASSWORD: qwer123!
      POSTGRES_DB: Breast_Cancer
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "myuser", "-d", "mydatabase"]
      interval: 10s
      timeout: 5s
      retries: 5

  data-generator:
    build:
      context: .
      dockerfile: data-generator.Dockerfile
    container_name: data-generator
    depends_on:
      postgres-server:
        condition: service_healthy
    command: ["postgres-server"]

networks:
  default:
    name: mlops-network
~~~

+ `test`: 테스트 할 명령어
+ `interval`: Healthcheck의 간격
+ `timeout`: Healthcheck의 timeout
+ `retries`: Timeout의 횟수
+ `networks:default`: 서비스 전체 기본 네트워크 수정

~~~bash
$ docker compose up -d
$ docker ps
CONTAINER ID   IMAGE                  COMMAND                  CREATED         STATUS                   PORTS                    NAMES
3181ea422214   part1_data-generator   "python data_generat…"   3 minutes ago   Up 2 minutes                                      data-generator
2afa679852c0   postgres:14.0          "docker-entrypoint.s…"   3 minutes ago   Up 3 minutes (healthy)   0.0.0.0:5432->5432/tcp   postgres-server
$ docker network ls
NETWORK ID     NAME            DRIVER    SCOPE
e73fc3c8f819   bridge          bridge    local
614822033f17   host            host      local
53929fb95a9e   mlops-network   bridge    local
b2b2a93007db   none            null      local
$ docker network inspect mlops-network
[
    {
        "Name": "mlops-network",
        "Id": "53929fb95a9ec94fc9bc254de07b5b57e7fef22a20aec4ed7c8f31f2f186667e",
        "Created": "2023-01-22T16:13:57.366605716Z",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv6": false,
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.20.0.0/16",
                    "Gateway": "172.20.0.1"
                }
            ]
        },
        "Internal": false,
        "Attachable": false,
        "Ingress": false,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": false,
        "Containers": {
            "3a7e5b68841e9bc17cc8032befb6b79766733017727e0c9e89077f60302b0ac9": {
                "Name": "postgres-server",
                "EndpointID": "8481bdcf9858bd9c5ebd3fc0c377a2d9f257be4952bba22dcb5fa1157b97905c",
                "MacAddress": "02:42:ac:14:00:02",
                "IPv4Address": "172.20.0.2/16",
                "IPv6Address": ""
            },
            "6bc0f3c76070cf412ff6b645bf81b43e3ffbbee6e2ea0f7b16f5c99c3207969c": {
                "Name": "data-generator",
                "EndpointID": "cb72782d8d9fcb9f153c92330b52631dd9eee6118e09db53a2dcda4d598e8ddb",
                "MacAddress": "02:42:ac:14:00:03",
                "IPv4Address": "172.20.0.3/16",
                "IPv6Address": ""
            }
        },
        "Options": {},
        "Labels": {
            "com.docker.compose.network": "default",
            "com.docker.compose.project": "part1",
            "com.docker.compose.version": "2.7.0"
        }
    }
]
~~~

~~~bash Network 제거 시
$ docker network rm ${Contianer Name}
~~~

~~~bash In Local
$ PGPASSWORD=mypassword psql -h localhost -p 5432 -U myuser -d mydatabase
~~~

~~~bash In Container
$ docker exec -it data-generator /bin/bash
$ PGPASSWORD=mypassword psql -h postgres-server -p 5432 -U myuser -d mydatabase
~~~

~~~sql
mydatabase=# \du
                                   List of roles
 Role name |                         Attributes                         | Member of
-----------+------------------------------------------------------------+-----------
 myuser    | Superuser, Create role, Create DB, Replication, Bypass RLS | {}

mydatabase=# select * from iris_data;
id  |         timestamp          | sepal_length | sepal_width | petal_length | petal_width | target
-----+----------------------------+--------------+-------------+--------------+-------------+--------
   1 | 2023-01-22 16:14:12.932928 |            5 |         3.5 |          1.6 |         0.6 |      0
   2 | 2023-01-22 16:14:13.956385 |          5.7 |         3.8 |          1.7 |         0.3 |      0
   3 | 2023-01-22 16:14:14.981818 |          4.9 |         3.1 |          1.5 |         0.2 |      0
   4 | 2023-01-22 16:14:16.017551 |          6.2 |         3.4 |          5.4 |         2.3 |      2
   5 | 2023-01-22 16:14:17.043793 |          5.5 |         4.2 |          1.4 |         0.2 |      0
   6 | 2023-01-22 16:14:18.060276 |          6.7 |         3.1 |          5.6 |         2.4 |      2
   7 | 2023-01-22 16:14:19.088206 |          4.8 |         3.4 |          1.9 |         0.2 |      0
   8 | 2023-01-22 16:14:20.121014 |          6.4 |         2.8 |          5.6 |         2.2 |      2
   9 | 2023-01-22 16:14:21.153898 |          5.6 |           3 |          4.1 |         1.3 |      1
  10 | 2023-01-22 16:14:22.182861 |          4.9 |         3.6 |          1.4 |         0.1 |      0
  11 | 2023-01-22 16:14:23.215977 |          5.4 |         3.4 |          1.7 |         0.2 |      0
  12 | 2023-01-22 16:14:24.246465 |          6.9 |         3.1 |          5.4 |         2.1 |      2
~~~

Local과 컨테이너 모두 같은 데이터인 것을 확인 !

> Docker 종료

~~~bash
$ docker stop $(docker ps -a -q)
6bc0f3c76070
3a7e5b68841e
$ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
~~~

---

# Application

[MLOps_Breast_Cancer](https://github.com/Zerohertz/MLOps-Breast-Cancer)

## 기존 이미지 및 컨테이너 제거

~~~bash
$ docker images -a
REPOSITORY             TAG       IMAGE ID       CREATED         SIZE
part1_data-generator   latest    897b2f3f7c6a   47 hours ago    521MB
postgres               14.0      01b2dbb34042   15 months ago   354MB
$ docker rmi part1_data-generator
Untagged: part1_data-generator:latest
Deleted: sha256:897b2f3f7c6ad201ff33051b3e51ed742b4b87957c9b1bbdb509dff739aff29c
$ docker rmi 01b2
Untagged: postgres:14.0
Untagged: postgres@sha256:db927beee892dd02fbe963559f29a7867708747934812a80f83bff406a0d54fd
Deleted: sha256:01b2dbb34042401d41c879c3b3532bda44af385039ddab244ccea33129ede5f2
Deleted: sha256:aa5a02e5a42413340554eaf4e480ff29aa0995f52f20e6850aba1294ea446615
Deleted: sha256:8ae061f97e7c0b55a99090dc46b17758df8f22197cc94ee13d3e3c346962b70f
Deleted: sha256:b94e43f422193f8631fef6d1e5b34876fd84d4b0b646aefa104e6e74354e7653
Deleted: sha256:5487353946763ba4b066f43a522cd06d22e2b2d5318e3f4102fdeea2e848d415
Deleted: sha256:53a9d25c0b80a35e6f135427bd7cd983a911284c06841ea6dad3e6445028b61b
Deleted: sha256:e1ab686161dc960848dabbbdaa9d2ae7fca9a55b33600ab8e6f0bda8aca80696
Deleted: sha256:c308b67ccb07aee664eee958fd2f9e8cfeb9bb2063b15146dea636329d13a389
Deleted: sha256:8767f12d94d12ac03e5d90c9db4bbe9bc65244e1c68ad3baf2c337b01afce127
Deleted: sha256:c442d24e3e4937c8fcd1f8d08112808d0589982c6777a1945ef8155976ad8c1d
Deleted: sha256:5d427e4482cc96bd8892c0429ca2699523cfb832ff062e4b343cee5f1fdb8ecf
Deleted: sha256:511351e1183f7db013b6dfa5b21c9897ddde8c411c44b39fce84563b3fa010d1
Deleted: sha256:035686ad2d4445723e28236d4813e94cd29f43dbc2169e83e5154e4f2d10e685
Deleted: sha256:18acdb3e3c0dea8b20494b048b10ae7ab1f455ab91ce3d1e77745b5fff77d5fd
$ docker images -a
REPOSITORY   TAG       IMAGE ID   CREATED   SIZE
$ docker ps -a
CONTAINER ID   IMAGE                  COMMAND                  CREATED        STATUS                      PORTS     NAMES
6bc0f3c76070   part1_data-generator   "python data_generat…"   47 hours ago   Exited (143) 47 hours ago             data-generator
3a7e5b68841e   postgres:14.0          "docker-entrypoint.s…"   47 hours ago   Exited (0) 47 hours ago               postgres-server
$ docker rm 6bc0
6bc0
$ docker rm 3a7e
3a7e
$ docker ps -a
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
~~~

## DB 컨테이너 생성

~~~bash
$ docker run -d \
--name postgres-server \
-p 1234:5432 \
-e POSTGRES_USER=zerohertz \
-e POSTGRES_PASSWORD=qwer123! \
-e POSTGRES_DB=Breast_Cancer \
postgres:14.0
$ docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS          PORTS                              NAMES
5f00ff41fd11   postgres:14.0   "docker-entrypoint.s…"   24 seconds ago   Up 24 seconds   5432/tcp, 0.0.0.0:1234->5678/tcp   postgres-server
$ PGPASSWORD=qwer123! psql -h localhost -p 1234 -U zerohertz -d Breast_Cancer
psql (15.1, server 14.0 (Debian 14.0-1.pgdg110+1))
Type "help" for help.

Breast_Cancer=# \du
                                   List of roles
 Role name |                         Attributes                         | Member of
-----------+------------------------------------------------------------+-----------
 zerohertz | Superuser, Create role, Create DB, Replication, Bypass RLS | {}
~~~

## Data Generator 컨테이너 생성

~~~python
>>> import pandas as pd
>>> from sklearn.datasets import load_breast_cancer
>>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
>>> df = pd.concat([X, y], axis="columns")
>>> df.dtypes
mean radius                float64
mean texture               float64
mean perimeter             float64
mean area                  float64
mean smoothness            float64
mean compactness           float64
mean concavity             float64
mean concave points        float64
mean symmetry              float64
mean fractal dimension     float64
radius error               float64
texture error              float64
perimeter error            float64
area error                 float64
smoothness error           float64
compactness error          float64
concavity error            float64
concave points error       float64
symmetry error             float64
fractal dimension error    float64
worst radius               float64
worst texture              float64
worst perimeter            float64
worst area                 float64
worst smoothness           float64
worst compactness          float64
worst concavity            float64
worst concave points       float64
worst symmetry             float64
worst fractal dimension    float64
target                       int64
dtype: object
~~~

~~~python data_generator.py
import time
from argparse import ArgumentParser

import pandas as pd
import psycopg2
from sklearn.datasets import load_breast_cancer


def get_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    df = pd.concat([X, y], axis="columns")
    rename_rule = {
        'mean radius': 'Feature_A', 'mean texture': 'Feature_B', 'mean perimeter': 'Feature_C', 'mean area': 'Feature_D', 'mean smoothness': 'Feature_E', 'mean compactness': 'Feature_F', 'mean concavity': 'Feature_G', 'mean concave points': 'Feature_H', 'mean symmetry': 'Feature_I', 'mean fractal dimension': 'Feature_J', 'radius error': 'Feature_K', 'texture error': 'Feature_L', 'perimeter error': 'Feature_M', 'area error': 'Feature_N', 'smoothness error': 'Feature_O', 'compactness error': 'Feature_P', 'concavity error': 'Feature_Q', 'concave points error': 'Feature_R', 'symmetry error': 'Feature_S', 'fractal dimension error': 'Feature_T', 'worst radius': 'Feature_U', 'worst texture': 'Feature_V', 'worst perimeter': 'Feature_W', 'worst area': 'Feature_X', 'worst smoothness': 'Feature_Y', 'worst compactness': 'Feature_Z', 'worst concavity': 'Feature_AA', 'worst concave points': 'Feature_BB', 'worst symmetry': 'Feature_CC', 'worst fractal dimension': 'Feature_DD'
    }
    df = df.rename(columns=rename_rule)
    return df


def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Breast_Cancer_Data (  
        id SERIAL PRIMARY KEY,  
        timestamp timestamp,
        Feature_A float8,
        Feature_B float8,
        Feature_C float8,
        Feature_D float8,
        Feature_E float8,
        Feature_F float8,
        Feature_G float8,
        Feature_H float8,
        Feature_I float8,
        Feature_J float8,
        Feature_K float8,
        Feature_L float8,
        Feature_M float8,
        Feature_N float8,
        Feature_O float8,
        Feature_P float8,
        Feature_Q float8,
        Feature_R float8,
        Feature_S float8,
        Feature_T float8,
        Feature_U float8,
        Feature_V float8,
        Feature_W float8,
        Feature_X float8,
        Feature_Y float8,
        Feature_Z float8,
        Feature_AA float8,
        Feature_BB float8,
        Feature_CC float8,
        Feature_DD float8,  
        target int  
    );"""
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()


def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO Breast_Cancer_Data
        (timestamp, Feature_A, Feature_B, Feature_C, Feature_D, Feature_E, Feature_F, Feature_G, Feature_H, Feature_I, Feature_J, Feature_K, Feature_L, Feature_M, Feature_N, Feature_O, Feature_P, Feature_Q, Feature_R, Feature_S, Feature_T, Feature_U, Feature_V, Feature_W, Feature_X, Feature_Y, Feature_Z, Feature_AA, Feature_BB, Feature_CC, Feature_DD, target)
        VALUES (
            NOW(),
            {data.Feature_A},
            {data.Feature_B},
            {data.Feature_C},
            {data.Feature_D},
            {data.Feature_E},
            {data.Feature_F},
            {data.Feature_G},
            {data.Feature_H},
            {data.Feature_I},
            {data.Feature_J},
            {data.Feature_K},
            {data.Feature_L},
            {data.Feature_M},
            {data.Feature_N},
            {data.Feature_O},
            {data.Feature_P},
            {data.Feature_Q},
            {data.Feature_R},
            {data.Feature_S},
            {data.Feature_T},
            {data.Feature_U},
            {data.Feature_V},
            {data.Feature_W},
            {data.Feature_X},
            {data.Feature_Y},
            {data.Feature_Z},
            {data.Feature_AA},
            {data.Feature_BB},
            {data.Feature_CC},
            {data.Feature_DD},
            {data.target}
        );
    """
    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()


def generate_data(db_connect, df):
    for _ in range(50):
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    args = parser.parse_args()

    db_connect = psycopg2.connect(
        user="zerohertz",
        password="qwer123!",
        host=args.db_host,
        port=5432,
        database="Breast_Cancer",
    )
    create_table(db_connect)
    df = get_data()
    generate_data(db_connect, df)
~~~

~~~Docker data-generator.Dockerfile
FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install scikit-learn pandas psycopg2-binary

COPY data_generator.py data_generator.py

ENTRYPOINT ["python", "data_generator.py", "--db-host"]

CMD ["localhost"]
~~~

## Result

~~~yaml docker-compose.yaml
version: "3"

services:
  postgres-server:
    image: postgres:14.0
    container_name: postgres-server
    ports:
      - 1234:5432
    environment:
      POSTGRES_USER: zerohertz
      POSTGRES_PASSWORD: qwer123!
      POSTGRES_DB: Breast_Cancer
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "myuser", "-d", "mydatabase"]
      interval: 10s
      timeout: 5s
      retries: 5

  data-generator:
    build:
      context: .
      dockerfile: Dockerfile_data-generator
    container_name: data-generator
    depends_on:
      postgres-server:
        condition: service_healthy
    command: ["postgres-server"]

networks:
  default:
    name: mlops-network
~~~

~~~sql
$ PGPASSWORD=qwer123! psql -h localhost -p 1234 -U zerohertz -d Breast_Cancer
psql (15.1, server 14.0 (Debian 14.0-1.pgdg110+1))
Type "help" for help.

Breast_Cancer=# \d
                     List of relations
 Schema |           Name            |   Type   |   Owner
--------+---------------------------+----------+-----------
 public | breast_cancer_data        | table    | zerohertz
 public | breast_cancer_data_id_seq | sequence | zerohertz
(2 rows)

Breast_Cancer=# select * from breast_cancer_data;
 id |         timestamp          | feature_a | feature_b | feature_c | feature_d | feature_e | feature_f | feature_g | feature_h | feature_i | feature_j | feature_k | feature_l | feature_m | feature_n | feature_o | feature_p | feature_q | feature_r | feature_s | feature_t | feature_u | feature_v | feature_w | feature_x | feature_y | feature_z | feature_aa | feature_bb | feature_cc | feature_dd | target
----+----------------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------------+------------+------------+------------+--------
  1 | 2023-01-24 17:53:26.659024 |     11.57 |     19.04 |      74.2 |     409.7 |   0.08546 |   0.07722 |   0.05485 |   0.01428 |    0.2031 |   0.06267 |    0.2864 |      1.44 |     2.206 |      20.3 |  0.007278 |   0.02047 |   0.04447 |  0.008799 |   0.01868 |  0.003339 |     13.07 |     26.98 |     86.43 |     520.5 |    0.1249 |    0.1937 |      0.256 |    0.06664 |     0.3035 |    0.08284 |      1
  2 | 2023-01-24 17:53:27.682335 |     11.16 |     21.41 |     70.95 |     380.3 |    0.1018 |   0.05978 |  0.008955 |   0.01076 |    0.1615 |   0.06144 |    0.2865 |     1.678 |     1.968 |     18.99 |  0.006908 |  0.009442 |  0.006972 |  0.006159 |   0.02694 |   0.00206 |     12.36 |     28.92 |     79.26 |       458 |    0.1282 |    0.1108 |    0.03582 |    0.04306 |     0.2976 |    0.07123 |      1
  3 | 2023-01-24 17:53:28.704999 |     27.42 |     26.27 |     186.9 |      2501 |    0.1084 |    0.1988 |    0.3635 |    0.1689 |    0.2061 |   0.05623 |     2.547 |     1.306 |     18.65 |     542.2 |   0.00765 |   0.05374 |   0.08055 |   0.02598 |   0.01697 |  0.004558 |     36.04 |     31.37 |     251.2 |      4254 |    0.1357 |    0.4256 |     0.6833 |     0.2625 |     0.2641 |    0.07427 |      0
  4 | 2023-01-24 17:53:29.735666 |     12.18 |     14.08 |     77.25 |     461.4 |   0.07734 |   0.03212 |   0.01123 |  0.005051 |    0.1673 |   0.05649 |    0.2113 |    0.5996 |     1.438 |     15.82 |  0.005343 |  0.005767 |   0.01123 |  0.005051 |   0.01977 | 0.0009502 |     12.85 |     16.47 |      81.6 |     513.1 |    0.1001 |   0.05332 |    0.04116 |    0.01852 |     0.2293 |    0.06037 |      1
  5 | 2023-01-24 17:53:30.767395 |     17.01 |     20.26 |     109.7 |     904.3 |   0.08772 |   0.07304 |    0.0695 |    0.0539 |    0.2026 |   0.05223 |    0.5858 |    0.8554 |     4.106 |     68.46 |  0.005038 |   0.01503 |   0.01946 |   0.01123 |   0.02294 |  0.002581 |      19.8 |     25.05 |       130 |      1210 |    0.1111 |    0.1486 |     0.1932 |     0.1096 |     0.3275 |    0.06469 |      0
...
~~~

---

# Reference

+ [MLOps for MLE: 01. Database](https://mlops-for-mle.github.io/tutorial/docs/category/01-database)