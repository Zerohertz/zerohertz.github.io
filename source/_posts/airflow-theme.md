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

<img width="1912" alt="amelia.css" src="/images/airflow-theme/263471346-876e23f4-0fc1-4104-b5f5-0d39b08d7b6c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150156Z&X-Amz-Expires=300&X-Amz-Signature=8b96880519b854e01274fba5be7287dce154c075a97cfd21212eba19f61d613a&X-Amz-SignedHeaders=host">

> cerulean.css

<img width="1912" alt="cerulean.css" src="/images/airflow-theme/263471351-683f19d3-8855-42c4-80d0-de92fafcd128.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150157Z&X-Amz-Expires=300&X-Amz-Signature=23c06c71948fe0f194b4505f5b2bb6c643ee2435912aad40d38831d10060adb6&X-Amz-SignedHeaders=host">

> cosmo.css

<img width="1912" alt="cosmo.css" src="/images/airflow-theme/263471354-52bb9358-0aa4-4ef8-9cda-31ba4a3c2d8a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150159Z&X-Amz-Expires=300&X-Amz-Signature=ff0bde0a9fb67fe96e9e21780aa6aa1cb620ceb4e6948485602359362fab46ea&X-Amz-SignedHeaders=host">

> cyborg.css

<img width="1912" alt="cyborg.css" src="/images/airflow-theme/263471357-10d4733f-ad34-4aea-b9e4-6b11a55763e6.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150201Z&X-Amz-Expires=300&X-Amz-Signature=df1e912736769e58e4a4e7ffdc2120d03dbc6307bd889d2015eb24fc496a3b5b&X-Amz-SignedHeaders=host">

> darkly.css

<img width="1912" alt="darkly.css" src="/images/airflow-theme/263471358-f6a7f449-39f4-43ab-b591-a460f53ff0b1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150204Z&X-Amz-Expires=300&X-Amz-Signature=e69b8e0e013d3d0222141ab188f436513ef58d0314879744ca5c6a1a819c9742&X-Amz-SignedHeaders=host">

> flatly.css

<img width="1912" alt="flatly.css" src="/images/airflow-theme/263471360-b3fc752c-c157-41e9-9b55-8466a73d0ece.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150205Z&X-Amz-Expires=300&X-Amz-Signature=4544864e73e8931440acdcd09ad3ee20691610abe117541f0db81d5c6c84fac1&X-Amz-SignedHeaders=host">

> journal.css

<img width="1912" alt="journal.css" src="/images/airflow-theme/263471790-97a192d5-1b3f-46e4-905d-93303222b768.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150207Z&X-Amz-Expires=300&X-Amz-Signature=cb0899897b719dbbe4128f762766788d0232bea67a50227777794bbf91669581&X-Amz-SignedHeaders=host">

> lumen.css

<img width="1912" alt="lumen.css" src="/images/airflow-theme/263471363-b54f0a73-0e7f-47bc-83d3-d28b13865e30.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150209Z&X-Amz-Expires=300&X-Amz-Signature=98009823e1f214f64c2fba5d3b058a46e5e9ba43b5d80859afac9882088dbc3c&X-Amz-SignedHeaders=host">

> paper.css

<img width="1912" alt="paper.css" src="/images/airflow-theme/263471365-3846f68d-24d8-464c-870f-9e4742a60c72.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150211Z&X-Amz-Expires=300&X-Amz-Signature=4b95ec3e7a2b45efd583631c9e4d3de41829a5a5d1448dc634d4e5ab3888bd5c&X-Amz-SignedHeaders=host">

> readable.css

<img width="1912" alt="readable.css" src="/images/airflow-theme/263471366-c70a2333-8e01-4dab-990e-f1c1200e3e53.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150213Z&X-Amz-Expires=300&X-Amz-Signature=9f61a944881e119caeaf029ac47dc927fa3c3b30922ae34ced59d0dc15ccd0e8&X-Amz-SignedHeaders=host">

> sandstone.css

<img width="1912" alt="sandstone.css" src="/images/airflow-theme/263471367-fd0dfeb0-5c9c-4896-b11b-bc8ca42ffa96.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150215Z&X-Amz-Expires=300&X-Amz-Signature=0c91d8deadbbae007b1a9fd618a8ff6dc5478f2f867263b71207fe4c11411ac1&X-Amz-SignedHeaders=host">

> simplex.css

<img width="1912" alt="simplex.css" src="/images/airflow-theme/263471348-9d24a4d6-7939-4cef-b844-ceabde775f12.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150217Z&X-Amz-Expires=300&X-Amz-Signature=01b7a99150a74d99d6fcdffe983ef991c9aba4c25b8f99574d11a69c5883a5d0&X-Amz-SignedHeaders=host">

> slate.css

<img width="1912" alt="slate.css" src="/images/airflow-theme/263471878-25acde5e-a682-4c4d-b0d6-0cf3d846fedc.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150219Z&X-Amz-Expires=300&X-Amz-Signature=ecbc8dceb35f439cf4d76e13f1472a9c98721fe0d2f2b660b3f4019e331135a0&X-Amz-SignedHeaders=host">

> solar.css

<img width="1912" alt="solar.css" src="/images/airflow-theme/263471370-71d62b14-fe59-4cd5-a707-6535610f0a31.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150221Z&X-Amz-Expires=300&X-Amz-Signature=351ff335afe05a63be4db598796a21edd4bec44813943009cb75f97e2a94a4cb&X-Amz-SignedHeaders=host">

> spacelab.css

<img width="1912" alt="spacelab.css" src="/images/airflow-theme/263471372-966cc814-23f2-40d0-b78d-e2a9de18ae6c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150223Z&X-Amz-Expires=300&X-Amz-Signature=d5384bb8501d5bc5a7b7f4ed6227d538aa375291ede1c488cb1b63e7837b5a67&X-Amz-SignedHeaders=host">

> superhero.css

<img width="1912" alt="superhero.css" src="/images/airflow-theme/263471373-91c43702-6537-4999-adab-fcb3fe3a4522.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150225Z&X-Amz-Expires=300&X-Amz-Signature=d80ce54581c7e143f7e3fa7c43d40e16c9767ce04faaef10747ab2717337db3d&X-Amz-SignedHeaders=host">

> united.css

<img width="1912" alt="united.css" src="/images/airflow-theme/263471375-1463e9b9-b533-4333-b3d6-6b2a4b6c9d80.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150227Z&X-Amz-Expires=300&X-Amz-Signature=fd7c6fed7ba9dafd307c43e6b9504b408f43c2f6dfc8237cccbf0cfc716d6ec5&X-Amz-SignedHeaders=host">

> yeti.css

<img width="1912" alt="yeti.css" src="/images/airflow-theme/263471376-a2b1ffaf-99cc-4a68-9684-e944bb10a075.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20241116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241116T150229Z&X-Amz-Expires=300&X-Amz-Signature=84e94faf5cbc6938068976a07ad797a8f4ffb813c8a5b7b5fd2c7be0878597e0&X-Amz-SignedHeaders=host">