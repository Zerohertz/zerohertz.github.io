---
title: 'MLOps for MLE: Model Development'
date: 2023-01-28 01:01:35
categories:
- 4. MLOps
tags:
- Python
- scikit-learn
---
# Base Model Development

## 학습 및 평가 데이터 선정

~~~python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2023)
~~~

## 모델 개발 및 학습

~~~python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)
~~~

+ `sklearn.preprocessing.StandardScaler()`: Standardize features by removing the mean and scaling to unit variance
  + $z = (x - u) / s$
    + $u$: Mean of the training samples
    + $s$: Standard deviation of the training samples
  + `StandardScaler.fit_transform(X)`: Fit to data, then transform it
  + `StandardScaler.transform(X)`: Perform standardization by centering and scaling

<!-- More -->

~~~python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

classifier = SVC()
classifier.fit(scaled_X_train, y_train)

train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy [%]:", train_acc * 100) # Train Accuracy [%]: 98.24175824175823
print("Valid Accuracy [%]:", valid_acc * 100) # Valid Accuracy [%]: 97.36842105263158
~~~

+ `sklearn.svm.SVC()`: C-Support Vector Classification
  + `SVC.fit(X, y)`: Fit the SVM model according to the given training data
  + `SVC.predict(X)`: Number of support vectors for each class
+ `sklearn.metrics.accuracy_score(y_true, y_pred)`: Accuracy classification score

## 모델 파이프라인

~~~python
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc) # Train Accuracy : 0.9875
print("Valid Accuracy :", valid_acc) # Valid Accuracy : 0.95
~~~

+ `sklearn.pipeline.Pipeline(steps)`: Pipeline of transforms with a final estimator
  + `Pipeline.fit(X, y)`: Fit the model
  + `Pipeline.predict(X)`: Transform the data, and apply `predict` with the final estimator

## 학습된 모델 저장, 저장된 모델 불러오기

~~~python
import joblib

joblib.dump(model_pipeline, "Etc./db_pipeline.joblib")
~~~

+ `joblib.dump(value, filename)`: Persist an arbitrary Python object into one file

~~~python
pipeline_load = joblib.load("Etc./db_pipeline.joblib")
~~~

+ `joblib.load(filename)`: Reconstruct a Python object from a file persisted with `joblib.dump`

---

# Load Data from Database

~~~python
import pandas as pd
import psycopg2

# PGPASSWORD=qwer123! psql -h localhost -p 1234 -U zerohertz -d Breast_Cancer
db_connect = psycopg2.connect(host="localhost", database="Breast_Cancer", user="zerohertz", password="qwer123!", port="1234")
df = pd.read_sql("SELECT * FROM breast_cancer_data ORDER BY id DESC LIMIT 100", db_connect)
X, y = df.drop(["id", "timestamp", "target"], axis="columns"), df["target"]

df.to_csv("Etc./DB.csv", index=False)
~~~

+ `psycopg2.connect(dbname, user, password)`
  + `dbname`: The database name (`database` is a deprecated alias)
  + `user`: User name used to authenticate
  + `password`: Password used to authenticate
  + `host`: Database host address (defaults to UNIX socket if not provided)
  + `port`: Connection port number (defaults to 5432 if not provided)
+ `pandas.read_sql(sql, con)`: Read SQL query or database table into a `pandas.DataFrame`
+ `pandas.DataFrame.drop(label, axis)`: Drop specified labels from rows or columns
+ `pandas.DataFrame.to_csv(path_or_buf, index=True)`: Write object to a comma-separated values (csv) file

## PostgreSQL DB 기반 모델 훈련

~~~python
import joblib
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# PGPASSWORD=qwer123! psql -h localhost -p 1234 -U zerohertz -d Breast_Cancer
db_connect = psycopg2.connect(host="localhost", database="Breast_Cancer", user="zerohertz", password="qwer123!", port="1234")
df = pd.read_sql("SELECT * FROM breast_cancer_data ORDER BY id DESC LIMIT 100", db_connect)
X, y = df.drop(["id", "timestamp", "target"], axis="columns"), df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2023)

model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc) # Load Model Train Accuracy : 0.9875
print("Valid Accuracy :", valid_acc) # Load Model Valid Accuracy : 0.95

joblib.dump(model_pipeline, "Etc./db_pipeline.joblib")

df.to_csv("Etc./DB.csv", index=False)
~~~

## 모델 검증

~~~python
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Etc./DB.csv")
X, y = df.drop(["id", "timestamp", "target"], axis="columns"), df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2023)

pipeline_load = joblib.load("Etc./db_pipeline.joblib")

load_train_pred = pipeline_load.predict(X_train)
load_valid_pred = pipeline_load.predict(X_valid)

load_train_acc = accuracy_score(y_true=y_train, y_pred=load_train_pred)
load_valid_acc = accuracy_score(y_true=y_valid, y_pred=load_valid_pred)

print("Load Model Train Accuracy :", load_train_acc) # Load Model Train Accuracy : 0.9875
print("Load Model Valid Accuracy :", load_valid_acc) # Load Model Valid Accuracy : 0.95
~~~

---

# Reference

+ [MLOps for MLE: 02. Model Development](https://mlops-for-mle.github.io/tutorial/docs/category/02-model-development)