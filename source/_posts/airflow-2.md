---
title: Apache Airflow (2)
date: 2023-07-27 21:11:41
categories:
- 3. DevOps
tags:
- Airflow
- Python
---
# Templating tasks using the Airflow context

[Wikimedia](https://www.wikimedia.org/)에서 제공하는 API를 통해 데이터를 `.gz` 확장자로 받아오고 schedule에 따른 증분 데이터를 적재하여 DAG와 operator가 Airflow에서 어떻게 작동하는지, 그리고 workflow를 어떻게 schedule하는지 이해해보자.
우선 API를 통해 아래와 같이 원하는 기간의 데이터를 받을 수 있다.

```shell
$ wget https://dumps.wikimedia.org/other/pageviews/2023/2023-07/pageviews-20230726-010000.gz
--2023-07-27 21:16:24--  https://dumps.wikimedia.org/other/pageviews/2023/2023-07/pageviews-20230726-010000.gz
dumps.wikimedia.org (dumps.wikimedia.org) 해석 중... 208.80.154.142
다음으로 연결 중: dumps.wikimedia.org (dumps.wikimedia.org)|208.80.154.142|:443... 연결했습니다.
HTTP 요청을 보냈습니다. 응답 기다리는 중... 200 OK
길이: 44600005 (43M) [application/octet-stream]
저장 위치: `pageviews-20230726-010000.gz'

pageviews-20230726-010000.gz  100%[=================================================>]  42.53M  4.56MB/s    /  9.9s

2023-07-27 21:16:36 (4.28 MB/s) - `pageviews-20230726-010000.gz' 저장함 [44600005/44600005]
```

<!-- More -->

더 자세히 데이터를 살펴보기 위해 압축을 해제한 결과는 아래와 같다.

```yaml pageviews-20230726-010000
aa Main_Page 15 0
aa Special:Contributions/MF-Warburg 1 0
aa Special:UserLogin 2 0
aa User:JAnDbot 1 0
aa User:Litlok 1 0
aa Wikipedia:Babel 1 0
...
zu.m Umzulendle 1 0
zu.m Wikipedia:Umnyango_wamgwamanda 1 0
zu.m Winnie_Madikizela-Mandela 1 0
zu.m XVideos 1 0
zu.m.d Ikhasi_Elikhulu 1 0
```

해당 데이터의 의미는 왼쪽부터 도메인 코드, 페이지 제목, 조회수, 응답 크기 (byte)를 의미한다.
증분 데이터를 가져오기 위해서는 원하는 시점을 API에 호출할 수 있도록 구성 요소를 나눠야한다.
Wikimedia API의 구성 요소는 아래와 같다.

```shell
$ wget https://dumps.wikimedia.org/other/pageviews/{year}/{year}-{month}/pageviews-{year}{month}{day}-{hour}0000.gz
```

이렇게 구성 요소로 나눠진 API를 Airflow에서 `BashOperator`와 `PythonOperator`로 호출하여 원하는 시점의 데이터를 불러올 수 있다.

## BashOperator

`BashOperator`로 API를 호출하여 데이터를 저장하기 위해 API의 구성 요소를 입력하기 위해 Jinja template을 사용한다.
[Jinja](https://jinja.palletsprojects.com/)는 python으로 작성된 template engine이다.
아래와 같은 형식으로 Jinja를 통해 데이터를 쉽게 삽입하고 반복할 수 있다.

|Type|Template|Mean|
|:-:|:-:|:-:|
|변수 (Variables)|`{{ variable_name }}`|중괄호 두 개 안에 변수명을 입력하여 값 삽입|
|주석 (Comments)|`{` `# This is a comment #` `}`|Template에서 무시되는 주석 생성|
|표현식 (Expressions)|`{% expression %}`|반복문, 조건문, 함수 호출 등을 위해 사용|
|If문|`{% if condition %}`|조건문|
|For문|`{% for item in list %}`|반복문|
|While문|`{% while condition %}`|반복문|

Jinja template과 Airflow의 `execution_date`를 활용해서 API의 구성 요소를 아래와 같이 삽입하여 실행할 수 있다.

```python
import airflow
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def _print_context(**kwargs):
    print("=" * 100)
    for i, j in kwargs.items():
        print(i, ":\t", j)
    print("=" * 100)


@dag(
    dag_id="Chap04_1",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@hourly",
)
def Chap04():
    get_data = BashOperator(
        task_id="get_data",
        bash_command=(
            "curl -o /opt/airflow/data/wikipageviews.gz "
            "https://dumps.wikimedia.org/other/pageviews/"
            "{{ execution_date.year }}/"
            "{{ execution_date.year }}-{{ '{:02}'.format(execution_date.month) }}/"
            "pageviews-{{ execution_date.year }}"
            "{{ '{:02}'.format(execution_date.month) }}"
            "{{ '{:02}'.format(execution_date.day) }}-"
            "{{ '{:02}'.format(execution_date.hour) }}0000.gz"
        ),
    )
    """
    NOTE: execution_date를 통한 API 호출
    """

    print_context = PythonOperator(
        task_id="print_context",
        python_callable=_print_context,
    )
    """
    NOTE: Task Context 출력
    """

    get_data >> print_context


DAG = Chap04()
```

실행 결과 아래와 같이 `execution_date`의 각 변수가 잘 삽입되어 API를 호출했음을 확인할 수 있다.

![bashoperator](/images/airflow-2/bashoperator.png)


Airflow에서 `execution_date`와 같은 변수를 task context라 칭한다.
어떤 task context 변수가 존재하는지 알아보기 위해 `print_context` task를 실행하였고 결과는 아래와 같다.

<details>
<summary>
<code>print_context</code>의 결과
</summary>

```bash
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ====================================================================================================
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - conf :	 <***.configuration.AirflowConfigParser object at 0xffffa7908810>
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - dag :	 <DAG: Chap04_1>
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - dag_run :	 <DagRun Chap04_1 @ 2023-07-26 11:00:00+00:00: scheduled__2023-07-26T11:00:00+00:00, state:running, queued_at: 2023-07-27 12:52:31.416749+00:00. externally triggered: False>
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - data_interval_end :	 2023-07-26T12:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - data_interval_start :	 2023-07-26T11:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ds :	 2023-07-26
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ds_nodash :	 20230726
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'execution_date' from the template is deprecated and will be removed in a future version. Please use 'data_interval_start' or 'logical_date' instead.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - execution_date :	 2023-07-26T11:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - expanded_ti_count :	 None
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - inlets :	 []
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - logical_date :	 2023-07-26T11:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - macros :	 <module '***.macros' from '/home/***/.local/lib/python3.7/site-packages/***/macros/__init__.py'>
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'next_ds' from the template is deprecated and will be removed in a future version. Please use '{{ data_interval_end | ds }}' instead.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - next_ds :	 2023-07-26
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'next_ds_nodash' from the template is deprecated and will be removed in a future version. Please use '{{ data_interval_end | ds_nodash }}' instead.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - next_ds_nodash :	 20230726
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'next_execution_date' from the template is deprecated and will be removed in a future version. Please use 'data_interval_end' instead.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - next_execution_date :	 2023-07-26T12:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - outlets :	 []
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - params :	 {}
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_data_interval_start_success :	 2023-07-26T05:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_data_interval_end_success :	 2023-07-26T06:00:00+00:00
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'prev_ds' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_ds :	 2023-07-26
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'prev_ds_nodash' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_ds_nodash :	 20230726
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'prev_execution_date' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_execution_date :	 2023-07-26T10:00:00+00:00
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'prev_execution_date_success' from the template is deprecated and will be removed in a future version. Please use 'prev_data_interval_start_success' instead.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_execution_date_success :	 2023-07-26T05:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - prev_start_date_success :	 2023-07-27T12:52:29.936814+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - run_id :	 scheduled__2023-07-26T11:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - task :	 <Task(PythonOperator): print_context>
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - task_instance :	 <TaskInstance: Chap04_1.print_context scheduled__2023-07-26T11:00:00+00:00 [running]>
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - task_instance_key_str :	 Chap04_1__print_context__20230726
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - test_mode :	 False
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ti :	 <TaskInstance: Chap04_1.print_context scheduled__2023-07-26T11:00:00+00:00 [running]>
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'tomorrow_ds' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - tomorrow_ds :	 2023-07-27
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'tomorrow_ds_nodash' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - tomorrow_ds_nodash :	 20230727
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - triggering_dataset_events :	 defaultdict(<class 'list'>, {})
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ts :	 2023-07-26T11:00:00+00:00
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ts_nodash :	 20230726T110000
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ts_nodash_with_tz :	 20230726T110000+0000
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - var :	 {'json': None, 'value': None}
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - conn :	 None
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'yesterday_ds' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - yesterday_ds :	 2023-07-25
[2023-07-27, 12:52:37 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.7/site-packages/***/utils/context.py:313: AirflowContextDeprecationWarning: Accessing 'yesterday_ds_nodash' from the template is deprecated and will be removed in a future version.
  warnings.warn(_create_deprecation_warning(k, replacements))

[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - yesterday_ds_nodash :	 20230725
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - templates_dict :	 None
[2023-07-27, 12:52:37 UTC] {logging_mixin.py:137} INFO - ====================================================================================================
```
</details>
<br />

각 변수에 대한 설명은 [여기](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html)에 상세하게 정리되어 있다. 

## PythonOperator

이러한 task context를 이용해 아래와 같이 `BashOperator`를 사용하지 않고 `PythonOperator`로 API를 호출할 수 있다.

```python
from urllib import request

import airflow
from airflow.decorators import dag
from airflow.operators.python import PythonOperator


def _get_data(execution_date):
    """
    NOTE: Template of PythonOperator
    """
    year, month, day, hour, *_ = execution_date.timetuple()
    url = (
        "https://dumps.wikimedia.org/other/pageviews/"
        f"{year}/{year}-{month:0>2}/"
        f"pageviews-{year}{month:0>2}{day:0>2}-{hour:0>2}0000.gz"
    )
    output_path = "/opt/airflow/data/wikipageviews.gz"
    request.urlretrieve(url, output_path)


def _print_context(**kwargs):
    print("=" * 100)
    for i, j in kwargs.items():
        print(i, ":\t", j)
    print("=" * 100)


@dag(
    dag_id="Chap04_2",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@hourly",
    max_active_runs=1,
)
def Chap04():
    get_data = PythonOperator(
        task_id="get_data",
        python_callable=_get_data,
    )

    print_context = PythonOperator(
        task_id="print_context",
        python_callable=_print_context,
    )

    get_data >> print_context


DAG = Chap04()
```

![pythonoperator](/images/airflow-2/pythonoperator.png)

`PythonOperator`는 python에서 `*args`, `**kwargs`와 같이 함수의 parameter를 입력 받을 수 있다.

```python
...
def _get_data(output_path, execution_date):
    year, month, day, hour, *_ = execution_date.timetuple()
    url = (
        "https://dumps.wikimedia.org/other/pageviews/"
        f"{year}/{year}-{month:0>2}/"
        f"pageviews-{year}{month:0>2}{day:0>2}-{hour:0>2}0000.gz"
    )
    request.urlretrieve(url, output_path)
...
@dag(
    dag_id="Chap04_3",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@hourly",
    max_active_runs=1,
)
def Chap04():
    get_data = PythonOperator(
        task_id="get_data",
        python_callable=_get_data,
        op_args=["/opt/airflow/data/wikipageviews.gz"],
    )
    """
    NOTE: Same as
    op_kwargs={"output_path": "/opt/airflow/data/wikipageviews.gz"}
    """
...
```

```python
...
def _get_data(year, month, day, hour, output_path):
    """
    NOTE: op_kwargs
    """
    url = (
        "https://dumps.wikimedia.org/other/pageviews/"
        f"{year}/{year}-{month:0>2}/"
        f"pageviews-{year}{month:0>2}{day:0>2}-{hour:0>2}0000.gz"
    )
    request.urlretrieve(url, output_path)
...
@dag(
    dag_id="Chap04_4",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@hourly",
    max_active_runs=1,
)
def Chap04():
    get_data = PythonOperator(
        task_id="get_data",
        python_callable=_get_data,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "hour": "{{ execution_date.hour }}",
            "output_path": "/opt/airflow/data/wikipageviews.gz",
        },
    )
    """
    NOTE: op_kwargs
    {
        "day": "24",
        "hour": "1",
        "month": "7",
        "output_path": "/opt/airflow/data/wikipageviews.gz",
        "year": "2023"
    }
    """
...
```

이렇게 Jinja template을 `PythonOperator`의 `op_kwargs`으로 활용할 수 있다.
증분 데이터의 적재를 위해 아래와 같이 압축을 해제하고 데이터를 추출할 수 있다.

```python
...
def _fetch_pageviews(pagenames):
    result = dict.fromkeys(pagenames, 0)
    with open("/opt/airflow/data/wikipageviews", "r") as f:
        for line in f:
            domain_code, page_title, view_counts, _ = line.split(" ")
            if domain_code == "en" and page_title in pagenames:
                result[page_title] = view_counts
    print(result)


@dag(
    dag_id="Chap04_5",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@hourly",
    max_active_runs=1,
)
def Chap04():
    get_data = PythonOperator(
        ...
    )

    extract_gz = BashOperator(
        task_id="extract_gz",
        bash_command="gunzip --force /opt/airflow/data/wikipageviews.gz",
    )
    """
    NOTE: gunzip
    """

    fetch_pageviews = PythonOperator(
        task_id="fetch_pageviews",
        python_callable=_fetch_pageviews,
        op_kwargs={
            "pagenames": {
                "Google",
                "Amazon",
                "Apple",
                "Microsoft",
                "Facebook",
            },
        },
    )
    """
    NOTE: Read unzipped files
    """

    get_data >> extract_gz >> fetch_pageviews
...
```

`BashOperator`로 압축을 해제하고 `PythonOperator`로 데이터를 읽어 출력한 결과는 아래와 같다.

```bash
[2023-07-27, 13:24:28 UTC] {logging_mixin.py:137} INFO - {'Microsoft': '153', 'Apple': '47', 'Amazon': '52', 'Google': '493', 'Facebook': '449'}
```

## PostgresOperator

위에서 추출한 데이터를 PostgreSQL로 전송하기 위해 `PostgresOperator`를 사용할 수 있다.
PostgreSQL을 사용하기 위해 우선 Postgres server container를 `docker-compose.yaml`에 추가한다.

```docker docker-compse.yaml
...
services:
  postgres-server:
    image: postgres:13
    container_name: ${container_name}
    ports:
      - ${POSTGRES_PORT}:${POSTGRES_PORT}
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    healthcheck:
      test: ["CMD", "pg_isready", "-q", "-U", "${POSTGRES_USER}", "-d", "${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    command: postgres -p ${POSTGRES_PORT}
...
```

Airflow와의 연결을 위해 `airflow connections` 명령어를 사용하고 Postgres server 내에 table을 생성한다.

```shell
$ docker exec -it ${airflow-worker} /bin/bash
$ airflow connections add postgres-server \
                --conn-type postgres \
                --conn-host postgres-server \
                --conn-port ${POSTGRES_PORT} \
                --conn-login ${POSTGRES_USER} \
                --conn-password ${POSTGRES_PASSWORD}
PGPASSWORD=${POSTGRES_PASSWORD} psql -h postgres-server -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -f create_table.sql
```

```sql create_table.sql
CREATE TABLE IF NOT EXISTS
${POSTGRES_DB} (
    pagename TEXT NOT NULL,
    pageviewcount INT NOT NULL,
    execution_date DATE NOT NULL
);
```

준비가 완료되면 DAG들을 아래와 같이 준비할 수 있다.

```python
...
from airflow.providers.postgres.operators.postgres import PostgresOperator
...
def _fetch_pageviews(pagenames, execution_date):
    result = dict.fromkeys(pagenames, 0)
    with open("/opt/airflow/data/wikipageviews", "r") as f:
        for line in f:
            domain_code, page_title, view_counts, _ = line.split(" ")
            if domain_code == "en" and page_title in pagenames:
                result[page_title] = view_counts
    with open("/opt/airflow/data/postgres_query.sql", "w") as f:
        for pagename, pageviewcount in result.items():
            f.write(
                "INSERT INTO {POSTGRES_DB} VALUES ("
                f"'{pagename}', {pageviewcount}, '{execution_date}'"
                ");\n"
            )
    """
    NOTE: Write SQL query
    """
...
@dag(
    dag_id="Chap04_6",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="@hourly",
    template_searchpath="/opt/airflow/data",
    max_active_runs=1,
)
def Chap04():
    init_data = BashOperator(
        task_id="init_data",
        bash_command="rm -rf /opt/airflow/data/wikipageviews.gz",
    )

    get_data = PythonOperator(
        ...
    )

    extract_gz = BashOperator(
        ...
    )

    fetch_pageviews = PythonOperator(
        task_id="fetch_pageviews",
        python_callable=_fetch_pageviews,
        op_kwargs={
            "pagenames": {
                "Google",
                "Amazon",
                "Apple",
                "Microsoft",
                "Facebook",
            },
        },
    )

    write_to_postgres = PostgresOperator(
        task_id="write_to_postgres",
        postgres_conn_id="postgres-server",
        sql="postgres_query.sql",
    )
    """
    NOTE: Write to postgresql
    """

    init_data >> get_data >> extract_gz >> fetch_pageviews >> write_to_postgres
...
```

`fetch_pageviews`에서 `print()` 대신 `postgres_query.sql`에 query문을 작성하는 것을 확인할 수 있다.
이렇게 작성된 query문은 `PostgresOperator`을 통해 Postgres server로 전송된다.

![postgresoperator](/images/airflow-2/postgresoperator.png)

Webserver 상에서는 잘 실행되는 것으로 보이지만, 해당 DAG가 Postgres server에 증분 데이터를 잘 적재하는지 확인하기 위해 아래의 query문을 실행했다.

```sql check.sql
SELECT x.pagename, x.hr AS "hour", x.average AS "average pageviews"
FROM (
    SELECT
        pagename,
        date_part('hour', execution_date) AS hr,
        AVG(pageviewcount) AS average,
        ROW_NUMBER() OVER (PARTITION BY pagename ORDER BY AVG(pageviewcount) DESC) AS row_number
    FROM boaz
    GROUP BY pagename, date_part('hour', execution_date)
) AS x
WHERE x.row_number = 1;
```

```shell
$ PGPASSWORD=${POSTGRES_PASSWORD} psql -h postgres-server -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "select * from ${POSTGRES_DB};"
 pagename  | pageviewcount | execution_date 
-----------+---------------+----------------
 Facebook  |           449 | 2023-07-26
 Amazon    |            52 | 2023-07-26
 Apple     |            47 | 2023-07-26
 Microsoft |           153 | 2023-07-26
 Google    |           493 | 2023-07-26
 Facebook  |           545 | 2023-07-26
 Amazon    |            40 | 2023-07-26
 Apple     |            47 | 2023-07-26
 Microsoft |           158 | 2023-07-26
 Google    |           545 | 2023-07-26
...
$ PGPASSWORD=${POSTGRES_PASSWORD} psql -h postgres-server -p ${POSTGRES_PORT} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -f check.sql
 pagename  | hour |  average pageviews   
-----------+------+----------------------
 Amazon    |    0 |  69.9090909090909091
 Apple     |    0 |  57.2272727272727273
 Facebook  |    0 | 845.1363636363636364
 Google    |    0 | 994.6363636363636364
 Microsoft |    0 | 215.6363636363636364
(5 rows)
```

증분 데이터가 Postgres server에 잘 적재되고 있음을 확인할 수 있다.