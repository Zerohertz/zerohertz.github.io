---
title: 'MLOps for MLE: Kafka'
date: 2023-02-27 11:11:14
categories:
- 4. MLOps
tags:
- Python
- Kafka
- Docker
---
# Kafka System

~~~Dockerfile connect.Dockerfile
FROM confluentinc/cp-kafka-connect:7.3.0

ENV CONNECT_PLUGIN_PATH="/usr/share/java,/usr/share/confluent-hub-components"

RUN confluent-hub install --no-prompt snowflakeinc/snowflake-kafka-connector:1.5.5 &&\
  confluent-hub install --no-prompt confluentinc/kafka-connect-jdbc:10.2.2 &&\
  confluent-hub install --no-prompt confluentinc/kafka-connect-json-schema-converter:7.3.0
~~~

<!-- More -->

~~~yaml kafka-docker-compose.yaml
version: "3"

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.3.0
    container_name: zookeeper
    ports:
      - 2181:2181
    environment:
      ZOOKEEPER_SERVER_ID: ${ZOOKEEPER_SERVER_ID}
      ZOOKEEPER_CLIENT_PORT: ${ZOOKEEPER_CLIENT_PORT}
  broker:
    image: confluentinc/cp-kafka:7.3.0
    container_name: broker
    depends_on:
      - zookeeper
    ports:
      - 9092:9092
    environment:
      KAFKA_BROKER_ID: ${KAFKA_BROKER_ID}
      KAFKA_ZOOKEEPER_CONNECT: ${KAFKA_ZOOKEEPER_CONNECT}
      KAFKA_ADVERTISED_LISTENERS: ${KAFKA_ADVERTISED_LISTENERS}
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: ${KAFKA_LISTENER_SECURITY_PROTOCOL_MAP}
      KAFKA_INTER_BROKER_LISTENER_NAME: ${KAFKA_INTER_BROKER_LISTENER_NAME}
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: ${KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR}
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: ${KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS}
  schema-registry:
    image: confluentinc/cp-schema-registry:7.3.0
    container_name: schema-registry
    depends_on:
      - broker
    environment:
      SCHEMA_REGISTRY_HOST_NAME: ${SCHEMA_REGISTRY_HOST_NAME}
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: ${SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS}
      SCHEMA_REGISTRY_LISTENERS: ${SCHEMA_REGISTRY_LISTENERS}
  connect:
    build:
      context: .
      dockerfile: connect.Dockerfile
    container_name: connect
    depends_on:
      - broker
      - schema-registry
    ports:
      - 8083:8083
    environment:
      CONNECT_BOOTSTRAP_SERVERS: ${CONNECT_BOOTSTRAP_SERVERS}
      CONNECT_REST_ADVERTISED_HOST_NAME: ${CONNECT_REST_ADVERTISED_HOST_NAME}
      CONNECT_GROUP_ID: ${CONNECT_GROUP_ID}
      CONNECT_CONFIG_STORAGE_TOPIC: ${CONNECT_CONFIG_STORAGE_TOPIC}
      CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR: ${CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR}
      CONNECT_OFFSET_STORAGE_TOPIC: ${CONNECT_OFFSET_STORAGE_TOPIC}
      CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR: ${CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR}
      CONNECT_STATUS_STORAGE_TOPIC: ${CONNECT_STATUS_STORAGE_TOPIC}
      CONNECT_STATUS_STORAGE_REPLICATION_FACTOR: ${CONNECT_STATUS_STORAGE_REPLICATION_FACTOR}
      CONNECT_KEY_CONVERTER: ${CONNECT_KEY_CONVERTER}
      CONNECT_VALUE_CONVERTER: ${CONNECT_VALUE_CONVERTER}
      CONNECT_VALUE_CONVERTER_SCHEMA_REGISTRY_URL: ${CONNECT_VALUE_CONVERTER_SCHEMA_REGISTRY_URL}

networks:
  default:
    name: mlops-network
    external: true
~~~

~~~bash
/MLOps-Breast-Cancer/Kafka$ docker compose -p kafka -f kafka-docker-compose.yaml up -d
/MLOps-Breast-Cancer/Kafka$ docker ps
CONTAINER ID   IMAGE                                   COMMAND                  CREATED          STATUS                             PORTS                                        NAMES
5fda9cc9cdc3   kafka-connect                           "/etc/confluent/dock…"   15 seconds ago   Up 13 seconds (health: starting)   0.0.0.0:8083->8083/tcp, 9092/tcp             connect
fcc48e59bc25   confluentinc/cp-schema-registry:7.3.0   "/etc/confluent/dock…"   15 seconds ago   Up 13 seconds                      8081/tcp                                     schema-registry
f7869897a87f   confluentinc/cp-kafka:7.3.0             "/etc/confluent/dock…"   15 seconds ago   Up 14 seconds                      0.0.0.0:9092->9092/tcp                       broker
37199baa9da6   confluentinc/cp-zookeeper:7.3.0         "/etc/confluent/dock…"   15 seconds ago   Up 14 seconds                      2888/tcp, 0.0.0.0:2181->2181/tcp, 3888/tcp   zookeeper
~~~

---

# Source Connector

~~~json source_connector.json
{
    "name": "postgres-source-connector",
    "config": {
        "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector",
        "connection.url": "jdbc:postgresql://postgres-server:5432/Breast_Cancer",
        "connection.user": "zerohertz",
        "connection.password": "qwer123!",
        "table.whitelist": "breast_cancer_data",
        "topic.prefix": "postgres-source-",
        "topic.creation.default.partitions": 1,
        "topic.creation.default.replication.factor": 1,
        "mode": "incrementing",
        "incrementing.column.name": "id",
        "tasks.max": 2,
        "transforms": "TimestampConverter",
        "transforms.TimestampConverter.type": "org.apache.kafka.connect.transforms.TimestampConverter$Value",
        "transforms.TimestampConverter.field": "timestamp",
        "transforms.TimestampConverter.format": "yyyy-MM-dd HH:mm:ss.S",
        "transforms.TimestampConverter.target.type": "string"
    }
}
~~~

~~~bash
/MLOps-Breast-Cancer/Kafka$ curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @source_connector.json
{"name":"postgres-source-connector","config":{"connector.class":"io.confluent.connect.jdbc.JdbcSourceConnector","connection.url":"jdbc:postgresql://postgres-server:5432/Breast_Cancer","connection.user":"zerohertz","connection.password":"qwer123!","table.whitelist":"breast_cancer_data","topic.prefix":"postgres-source-","topic.creation.default.partitions":"1","topic.creation.default.replication.factor":"1","mode":"incrementing","incrementing.column.name":"id","tasks.max":"2","transforms":"TimestampConverter","transforms.TimestampConverter.type":"org.apache.kafka.connect.transforms.TimestampConverter$Value","transforms.TimestampConverter.field":"timestamp","transforms.TimestampConverter.format":"yyyy-MM-dd HH:mm:ss.S","transforms.TimestampConverter.target.type":"string","name":"postgres-source-connector"},"tasks":[],"type":"source"}%
/MLOps-Breast-Cancer/Kafka$ curl -X GET http://localhost:8083/connectors
["postgres-source-connector"]%
/MLOps-Breast-Cancer/Kafka$ curl -X GET http://localhost:8083/connectors/postgres-source-connector
{"name":"postgres-source-connector","config":{"connector.class":"io.confluent.connect.jdbc.JdbcSourceConnector","transforms.TimestampConverter.target.type":"string","incrementing.column.name":"id","topic.creation.default.partitions":"1","connection.password":"qwer123!","transforms.TimestampConverter.field":"timestamp","tasks.max":"2","transforms.TimestampConverter.type":"org.apache.kafka.connect.transforms.TimestampConverter$Value","transforms":"TimestampConverter","transforms.TimestampConverter.format":"yyyy-MM-dd HH:mm:ss.S","table.whitelist":"breast_cancer_data","mode":"incrementing","topic.prefix":"postgres-source-","connection.user":"zerohertz","topic.creation.default.replication.factor":"1","name":"postgres-source-connector","connection.url":"jdbc:postgresql://postgres-server:5432/Breast_Cancer"},"tasks":[{"connector":"postgres-source-connector","task":0}],"type":"source"}%
$ kcat -L -b localhost:9092
...
  topic "postgres-source-breast_cancer_data" with 1 partitions:
    partition 0, leader 1, replicas: 1, isrs: 1
...
$ kcat -b localhost:9092 -t postgres-source-breast_cancer_data
~~~

<img width="500" src="https://user-images.githubusercontent.com/42334717/221479676-bb9cb23e-76d1-4c36-8e27-affb2da42d18.gif">

---

# Sink Connector

~~~python create_table.py
import os
from dotenv import load_dotenv
import psycopg2


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
~~~

~~~Docker target.Dockerfile
FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install psycopg2-binary &&\
    pip install python-dotenv

COPY create_table.py create_table.py
COPY .env .env

ENTRYPOINT ["python", "create_table.py"]
~~~

~~~yaml target-docker-compose.yaml
version: "3"

services:
  target-postgres-server:
    image: postgres:14.0
    container_name: ${POSTGRES_HOST}
    ports:
      - 5433:5432
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "${POSTGRES_USER}", "-d", "${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  table-creator:
    build:
      context: .
      dockerfile: target.Dockerfile
    container_name: table-creator
    depends_on:
      target-postgres-server:
        condition: service_healthy

networks:
  default:
    name: mlops-network
    external: true
~~~

~~~bash
/MLOps-Breast-Cancer/Kafka$ docker compose -p target -f target-docker-compose.yaml up -d
/MLOps-Breast-Cancer/Kafka$ docker ps
CONTAINER ID   IMAGE                                   COMMAND                  CREATED             STATUS                       PORTS                                        NAMES
fb93165c1c87   postgres:14.0                           "docker-entrypoint.s…"   18 seconds ago      Up 17 seconds (healthy)      0.0.0.0:5432->5432/tcp                       target-postgres-server
~~~

~~~json sink_connector.json
{
    "name": "postgres-sink-connector",
    "config": {
        "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
        "connection.url": "jdbc:postgresql://target-postgres-server:5432/targetdatabase",
        "connection.user": "zerohertz_target",
        "connection.password": "qwer123!",
        "table.name.format": "breast_cancer_data",
        "topics": "postgres-source-breast_cancer_data",
        "auto.create": false,
        "auto.evolve": false,
        "tasks.max": 2,
        "transforms": "TimestampConverter",
        "transforms.TimestampConverter.type": "org.apache.kafka.connect.transforms.TimestampConverter$Value",
        "transforms.TimestampConverter.field": "timestamp",
        "transforms.TimestampConverter.format": "yyyy-MM-dd HH:mm:ss.S",
        "transforms.TimestampConverter.target.type": "Timestamp"
    }
}
~~~

~~~bash
/MLOps-Breast-Cancer/Kafka$ curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @sink_connector.json
{"name":"postgres-sink-connector","config":{"connector.class":"io.confluent.connect.jdbc.JdbcSinkConnector","connection.url":"jdbc:postgresql://target-postgres-server:5432/targetdatabase","connection.user":"zerohertz_target","connection.password":"qwer123!","table.name.format":"breast_cancer_data","topics":"postgres-source-breast_cancer_data","auto.create":"false","auto.evolve":"false","tasks.max":"2","transforms":"TimestampConverter","transforms.TimestampConverter.type":"org.apache.kafka.connect.transforms.TimestampConverter$Value","transforms.TimestampConverter.field":"timestamp","transforms.TimestampConverter.format":"yyyy-MM-dd HH:mm:ss.S","transforms.TimestampConverter.target.type":"Timestamp","name":"postgres-sink-connector"},"tasks":[],"type":"sink"}%
/MLOps-Breast-Cancer/Kafka$ curl -X GET http://localhost:8083/connectors
["postgres-sink-connector","postgres-source-connector"]%
/MLOps-Breast-Cancer/Kafka$ curl -X GET http://localhost:8083/connectors/postgres-sink-connector
{"name":"postgres-sink-connector","config":{"connector.class":"io.confluent.connect.jdbc.JdbcSinkConnector","transforms.TimestampConverter.target.type":"Timestamp","table.name.format":"breast_cancer_data","connection.password":"qwer123!","transforms.TimestampConverter.field":"timestamp","topics":"postgres-source-breast_cancer_data","tasks.max":"2","transforms.TimestampConverter.type":"org.apache.kafka.connect.transforms.TimestampConverter$Value","transforms":"TimestampConverter","transforms.TimestampConverter.format":"yyyy-MM-dd HH:mm:ss.S","auto.evolve":"false","connection.user":"zerohertz_target","name":"postgres-sink-connector","auto.create":"false","connection.url":"jdbc:postgresql://target-postgres-server:5432/targetdatabase"},"tasks":[{"connector":"postgres-sink-connector","task":0},{"connector":"postgres-sink-connector","task":1}],"type":"sink"}%
/MLOps-Breast-Cancer/Kafka$ PGPASSWORD=qwer123! psql -h localhost -p 5433 -U zerohertz_target -d targetdatabase
~~~

<img width="500" src="https://user-images.githubusercontent.com/42334717/221571082-adb84ed7-fd46-4845-a01f-5bc4cf6ff665.gif">

---

# Reference

+ [MLOps for MLE: 07. Kafka](https://mlops-for-mle.github.io/tutorial/docs/category/07-kafka)