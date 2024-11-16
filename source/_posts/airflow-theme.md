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

<img src="/images/airflow-theme/amelia.css.png" alt="amelia.css" width="1912" />

> cerulean.css

<img src="/images/airflow-theme/cerulean.css.png" alt="cerulean.css" width="1912" />

> cosmo.css

<img src="/images/airflow-theme/cosmo.css.png" alt="cosmo.css" width="1912" />

> cyborg.css

<img src="/images/airflow-theme/cyborg.css.png" alt="cyborg.css" width="1912" />

> darkly.css

<img src="/images/airflow-theme/darkly.css.png" alt="darkly.css" width="1912" />

> flatly.css

<img src="/images/airflow-theme/flatly.css.png" alt="flatly.css" width="1912" />

> journal.css

<img src="/images/airflow-theme/journal.css.png" alt="journal.css" width="1912" />

> lumen.css

<img src="/images/airflow-theme/lumen.css.png" alt="lumen.css" width="1912" />

> paper.css

<img src="/images/airflow-theme/paper.css.png" alt="paper.css" width="1912" />

> readable.css

<img src="/images/airflow-theme/readable.css.png" alt="readable.css" width="1912" />

> sandstone.css

<img src="/images/airflow-theme/sandstone.css.png" alt="sandstone.css" width="1912" />

> simplex.css

<img src="/images/airflow-theme/simplex.css.png" alt="simplex.css" width="1912" />

> slate.css

<img src="/images/airflow-theme/slate.css.png" alt="slate.css" width="1912" />

> solar.css

<img src="/images/airflow-theme/solar.css.png" alt="solar.css" width="1912" />

> spacelab.css

<img src="/images/airflow-theme/spacelab.css.png" alt="spacelab.css" width="1912" />

> superhero.css

<img src="/images/airflow-theme/superhero.css.png" alt="superhero.css" width="1912" />

> united.css

<img src="/images/airflow-theme/united.css.png" alt="united.css" width="1912" />

> yeti.css

<img src="/images/airflow-theme/yeti.css.png" alt="yeti.css" width="1912" />