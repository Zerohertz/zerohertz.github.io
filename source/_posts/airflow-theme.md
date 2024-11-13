---
title: 'Apache Airflow: Theme'
date: 2023-08-26 23:57:21
categories:
- 3. DevOps
tags:
- Airflow
- Kubernetes
- Home Server
---
# Introduction

Apache Airflow에 적용 가능한 theme를 [공식 repository](https://github.com/apache/airflow/blob/main/airflow/config_templates/default_webserver_config.py)에서 아래와 같이 확인할 수 있다.

```python airflow/config_templates/default_webserver_config.py
# ----------------------------------------------------
# Theme CONFIG
# ----------------------------------------------------
# Flask App Builder comes up with a number of predefined themes
# that you can use for Apache Airflow.
# http://flask-appbuilder.readthedocs.io/en/latest/customizing.html#changing-themes
# Please make sure to remove "navbar_color" configuration from airflow.cfg
# in order to fully utilize the theme. (or use that property in conjunction with theme)
APP_THEME = "bootstrap-theme.css"  # default bootstrap
APP_THEME = "amelia.css"
APP_THEME = "cerulean.css"
APP_THEME = "cosmo.css"
APP_THEME = "cyborg.css"
APP_THEME = "darkly.css"
APP_THEME = "flatly.css"
APP_THEME = "journal.css"
APP_THEME = "lumen.css"
APP_THEME = "paper.css"
APP_THEME = "readable.css"
APP_THEME = "sandstone.css"
APP_THEME = "simplex.css"
APP_THEME = "slate.css"
APP_THEME = "solar.css"
APP_THEME = "spacelab.css"
APP_THEME = "superhero.css"
APP_THEME = "united.css"
APP_THEME = "yeti.css"
```

<!-- More -->

---

# Themes

Helm을 통해 Apache Airflow를 사용하면 theme를 아래와 같이 설정할 수 있다.

```yaml values.yaml
...
webserver:
  ...
  webserverConfig: |
    APP_THEME = "simplex.css"
...
```

> amelia.css

<img width="1912" alt="amelia.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/876e23f4-0fc1-4104-b5f5-0d39b08d7b6c">

> cerulean.css

<img width="1912" alt="cerulean.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/683f19d3-8855-42c4-80d0-de92fafcd128">

> cosmo.css

<img width="1912" alt="cosmo.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/52bb9358-0aa4-4ef8-9cda-31ba4a3c2d8a">

> cyborg.css

<img width="1912" alt="cyborg.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/10d4733f-ad34-4aea-b9e4-6b11a55763e6">

> darkly.css

<img width="1912" alt="darkly.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/f6a7f449-39f4-43ab-b591-a460f53ff0b1">

> flatly.css

<img width="1912" alt="flatly.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/b3fc752c-c157-41e9-9b55-8466a73d0ece">

> journal.css

<img width="1912" alt="journal.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/97a192d5-1b3f-46e4-905d-93303222b768">

> lumen.css

<img width="1912" alt="lumen.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/b54f0a73-0e7f-47bc-83d3-d28b13865e30">

> paper.css

<img width="1912" alt="paper.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/3846f68d-24d8-464c-870f-9e4742a60c72">

> readable.css

<img width="1912" alt="readable.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/c70a2333-8e01-4dab-990e-f1c1200e3e53">

> sandstone.css

<img width="1912" alt="sandstone.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/fd0dfeb0-5c9c-4896-b11b-bc8ca42ffa96">

> simplex.css

<img width="1912" alt="simplex.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/9d24a4d6-7939-4cef-b844-ceabde775f12">

> slate.css

<img width="1912" alt="slate.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/25acde5e-a682-4c4d-b0d6-0cf3d846fedc">

> solar.css

<img width="1912" alt="solar.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/71d62b14-fe59-4cd5-a707-6535610f0a31">

> spacelab.css

<img width="1912" alt="spacelab.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/966cc814-23f2-40d0-b78d-e2a9de18ae6c">

> superhero.css

<img width="1912" alt="superhero.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/91c43702-6537-4999-adab-fcb3fe3a4522">

> united.css

<img width="1912" alt="united.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/1463e9b9-b533-4333-b3d6-6b2a4b6c9d80">

> yeti.css

<img width="1912" alt="yeti.css" src="https://github.com/Zerohertz/Zerohertz/assets/42334717/a2b1ffaf-99cc-4a68-9684-e944bb10a075">