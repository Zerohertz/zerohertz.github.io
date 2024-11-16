---
title: 'MLOps for MLE: Stream'
date: 2023-03-09 13:31:03
categories:
- 4. MLOps
tags:
- Python
- FastAPI
- Kafka
- Docker
- Grafana
---
# Stream Serving

## Data Subscriber

```python data_subscriber.py
import os
from dotenv import load_dotenv

from json import loads

import psycopg2
import requests
from kafka import KafkaConsumer


def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS breast_cancer_prediction (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        breast_cancer_class int
    );"""
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()


def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO breast_cancer_prediction
        (timestamp, breast_cancer_class)
        VALUES (
            '{data["timestamp"]}',
            {data["target"]}
        );"""
    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()


def subscribe_data(db_connect, consumer):
    for msg in consumer:
        print(
            f"Topic : {msg.topic}\n"
            f"Partition : {msg.partition}\n"
            f"Offset : {msg.offset}\n"
            f"Key : {msg.key}\n"
            f"Value : {msg.value}\n",
        )

        msg.value["payload"].pop("id")
        msg.value["payload"].pop("target")
        ts = msg.value["payload"].pop("timestamp")

        response = requests.post(
            url="http://api-with-model:8000/predict",
            json=msg.value["payload"],
            headers={"Content-Type": "application/json"},
        ).json()
        response["timestamp"] = ts
        insert_data(db_connect, response)


if __name__ == "__main__":
    load_dotenv()
    db_connect = psycopg2.connect(
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        host=os.environ.get("POSTGRES_HOST"),
        port=5432,
        database=os.environ.get("POSTGRES_DB"),
    )
    create_table(db_connect)

    consumer = KafkaConsumer(
        "postgres-source-breast_cancer_data",
        bootstrap_servers="broker:29092",
        auto_offset_reset="earliest",
        group_id="breast_cancer_data-consumer-group",
        value_deserializer=lambda x: loads(x),
    )
    subscribe_data(db_connect, consumer)
```

<!-- More -->

## Docker Compose

```Docker stream.Dockerfile
FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install psycopg2-binary kafka-python requests\
    pip install python-dotenv

COPY data_subscriber.py data_subscriber.py
COPY .env .env

ENTRYPOINT ["python", "data_subscriber.py"]
```

```yaml stream-docker-compose.yaml
version: "3"

services:
  data-subscriber:
    build:
      context: .
      dockerfile: stream.Dockerfile
    container_name: data-subscriber

networks:
  default:
    name: mlops-network
    external: true
```

```bash stream.sh
docker compose -p kafka down
docker compose -p target down
docker compose -p stream down

cd ../Database
docker compose up -d
cd ../FastAPI
docker compose up -d
cd ../Kafka
docker compose -p kafka -f kafka-docker-compose.yaml up -d
docker compose -p target -f target-docker-compose.yaml up -d
curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @source_connector.json
curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @sink_connector.json
cd ../Stream
cp ../Kafka/.env ./
docker compose -p stream -f stream-docker-compose.yaml up -d
```

```bash
/MLOps-Breast-Cancer/Stream$ sh stream.sh
```

<img src="/images/mlops-for-mle-stream/containers.png" alt="containers" width="1571" />

> Container: data-subscriber

![data-subscriber](/images/mlops-for-mle-stream/data-subscriber.gif)


```bash
/MLOps-Breast-Cancer/Stream$ PGPASSWORD=qwer123! psql -h localhost -p 5433 -U zerohertz_target -d targetdatabase
```

```sql
psql (15.1, server 14.0 (Debian 14.0-1.pgdg110+1))
Type "help" for help.

targetdatabase=# \d
                           List of relations
 Schema |              Name               |   Type   |      Owner
--------+---------------------------------+----------+------------------
 public | breast_cancer_data              | table    | zerohertz_target
 public | breast_cancer_data_id_seq       | sequence | zerohertz_target
 public | breast_cancer_prediction        | table    | zerohertz_target
 public | breast_cancer_prediction_id_seq | sequence | zerohertz_target
(4 rows)
targetdatabase=# SELECT * FROM breast_cancer_prediction LIMIT 100;
```

![breast-cancer-prediction](/images/mlops-for-mle-stream/breast-cancer-prediction.gif)

---

# Dashboard

## Grafana Setup

```yaml grafana-docker-compose.yaml
version: "3"

services:
  grafana-dashboard:
    image: grafana/grafana
    ports:
     - 3000:3000
    environment:
      GF_SECURITY_ADMIN_USER: dashboarduser
      GF_SECURITY_ADMIN_PASSWORD: dashboardpassword
      GF_DASHBOARDS_MIN_REFRESH_INTERVAL: 500ms

networks:
  default:
    name: mlops-network
    external: true
```

```bash
/MLOps-Breast-Cancer/Stream$ docker compose -p dashboard -f grafana-docker-compose.yaml up -d
```

<img src="/images/mlops-for-mle-stream/grafana.png" alt="grafana" width="1202" />

## Grafana Dashboard

<img src="/images/mlops-for-mle-stream/grafana-dashboard.png" alt="grafana-dashboard" width="1202" />

## Source Database

<img src="/images/mlops-for-mle-stream/source-database.png" alt="source-database" width="1202" />

+ Name: `Source-database`
+ Host: `postgres-server:5432`
+ Database: `Breast_Cancer`
+ User: `zerohertz`
+ Password: `qwer123!`
+ TLS/SSL Mode: `disable`
+ Version: `14.0`

<img src="/images/mlops-for-mle-stream/dashboard.png" alt="dashboard" width="1202" />

## Inference Database

<img src="/images/mlops-for-mle-stream/inf.png" alt="inf" width="1233" />

+ Name: `Inference-database`
+ Host: `target-postgres-server:5432`
+ Database: `targetdatabase`
+ User: `zerohertz_target`
+ Password: `qwer123!`
+ TLS/SSL Mode: `disable`
+ Version: `14.0`

## Result

![result](/images/mlops-for-mle-stream/result.gif)

---

# Reference

+ [MLOps for MLE: 08. Stream](https://mlops-for-mle.github.io/tutorial/docs/category/08-stream)