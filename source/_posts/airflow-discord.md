---
title: Apache Airflow 기반 Discord Webhook 개발
date: 2023-08-13 21:03:10
categories:
- 3. DevOps
tags:
- Airflow
- Python
- Docker
- Kubernetes
- Home Server
---
# Introduction

[Fail2Ban](https://zerohertz.github.io/home-server-init/#Fail2Ban)의 명령어 중 `sudo fail2ban-client status sshd`을 사용하면 아래와 같은 결과가 나온다.

![jail](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/260302422-589245b2-4509-4711-be99-fbc844a11ad3.gif)

```shell
$ sudo fail2ban-client status sshd
Status for the jail: sshd                                           # Service에 대한 보호 규칙
|- Filter                                                           # 악의적인 활동 정보
|  |- Currently failed: 2                                           # 2개 기기의 잘못된 로그인 시도
|  |- Total failed:     7                                           # 시작된 이후 총 7번의 잘못된 로그인 시도
|  `- Journal matches:  _SYSTEMD_UNIT=sshd.service + _COMM=sshd     # Fail2Ban의 시스템 로그를 감시 방법
`- Actions                                                          # 악의적인 활동을 감지했을 때 취하는 조치에 대한 정보
   |- Currently banned: 1                                           # 현재 차단된 IP 주소의 수
   |- Total banned:     1                                           # 총 차단된 IP 주소의 수
   `- Banned IP list:   XXX.XXX.XXX.XXX                             # 차단된 IP
```

차단된 기기에서 SSH를 연결하려 시도하면 아래와 같이 실패하게 된다.

```shell
$ ssh ${USER}@${IP} -p ${PORT}
ssh: connect to host ${IP} port ${PORT}: Connection refused
```

이걸 매번 확인할 수는 없기 때문에 의문의 중국 해커가 공격하면 알람을 받을 수 있게 Apache Airflow와 Discord를 사용해보겠다!
~~사실 그거 안다해도 포맷 말고 할 수 있는게 없긴 한...~~

<!-- More -->

---

# Setup

Test 환경을 구축하기 위해 Conda를 설치한다.

```shell
$ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
$ bash Anaconda3-2021.05-Linux-x86_64.sh
$ source /home/${USER}/anaconda3/bin/activate
$ conda init
$ source ~/.bashrc
$ conda env list
# conda environments:
#
base                  *  /home/${USER}/anaconda3
$ conda create -n test python=3.8 -y
$ conda activate test
```

Discord에서 Webhook과 Bot의 차이는 아래와 같다.

|항목|Webhook|Bot|
|:-:|:-:|:-:|
|목적|특정 이벤트에 따른 자동 메시지 전송|사용자와의 상호작용 및 다양한 작업 수행|
|상호작용|없음<br />(Only 메시지 전송)|사용자와 상호작용 가능<br />(명령어 처리, 메시지 반응 등)|
|인증|웹훅 URL을 통한 인증|토큰을 사용한 인증|
|설정 및 관리|Discord 서버의 특정 채널에서 설정 및 관리|Discord Developer Portal에서 생성 및 관리<br />다양한 권한 설정 가능<br />여러 서버에 추가 가능|

진행하려는 service의 목표는 event 발생 시 Discord로 단순 메시지를 전송하는 것이므로 Webhook를 사용한다.
아래와 같이 Discord에서 Webhook을 발급받는다.

![discord](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/260303443-07d39198-b66c-4bee-a89a-b8e510bb46e7.png)

`웹후크 URL 복사` 버튼을 누르면 끝이다!
아래와 같이 테스트를 진행했다.

```shell
$ pip install requests
```

```python
import requests
import json

def send_discord_message(webhook_url, content):
    data = {
        "content": content
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(webhook_url, data=json.dumps(data), headers=headers)
    return response

>>> webhook_url = "${WEBHOOK_URL}"
>>> message = "Hello, World!"
>>> response
<Response [204]>
```

![test](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/260304378-46424e32-6c2b-41a8-b993-cad930ad5be5.png)

아주 잘 수신이 되는 것을 확인했다.

---

# DAG

1. `BashOperator`
   1. `sudo fail2ban-client status sshd` 실행
   2. 결과를 다음 task에 전달
2. `PythonOperator`
   1. 이전 단계의 결과와 적재된 결과 비교
   2. 다르다면 log 파일을 작성하고 Webhook으로 메시지 전달

하지만 `sudo fail2ban-client status sshd` 명령어는 pod 바깥의 사용자 계정에서 실행되어야 한다.
따라서 `BashOperator`를 사용하는 것이 의미가 없다.
`KubernetesPodOperator`을 사용해서 `/var/log/fail2ban.log`가 `/opt/airflow/logs/fail2ban.log`에 마운트 되도록 구성해보자!

## `KubernetesPodOperator`

그 전에 `KubernetesPodOperator`를 알아보고 예제를 살펴보자. (`PythonOperater` 등이 실행되는 pod의 기본 설정은 [여기](https://github.com/apache/airflow/blob/main/chart/files/pod-template-file.kubernetes-helm-yaml)를 참고)

+ Docker Image
  + 지정된 Docker image를 사용하여 작업 실행
  + 커스텀 로직이나 의존성을 가진 코드를 실행할 때 유용
+ Dynamic Environment
  + Kubernetes를 사용하여 작업별로 독립된 환경 제공
  + 각 작업의 독립적인 환경 실행 보장
  + 서로 다른 작업들 사이에서 의존성 충돌 회피 가능
+ Parameterization
  + 다양한 파라미터를 통해 pod의 설정 제어
  + Pod의 resource 제한, environment variables, volume mounts, ...
+ XCom Integration
  + XCom을 통해 데이터를 Airflow의 다른 작업과 공유
  + `/airflow/xcom/return.json` 경로에 데이터를 작성함으로써 XCom 값을 반환
+ Logging
  + `get_logs` 파라미터를 사용하여 pod에서 생성된 로그를 Airflow UI에 출력
+ Resource Management
  + Pod의 생성과 제거, 그리고 에러 핸들링 등의 lifecycle을 Airflow에서 관리

```python test.py
import json

import airflow
import requests
from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)


def _print_contexts(**kwargs):
    print("=" * 100)
    ti = kwargs["ti"]
    result_from_kubernetes_pod_operator = ti.xcom_pull(
        task_ids="test_kubernetes_pod_operator"
    )
    print("result_from_kubernetes_pod_operator:\t", result_from_kubernetes_pod_operator)
    print("=" * 100)


@dag(
    dag_id="test",
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval="@hourly",
    max_active_runs=1,
    catchup=False,
)
def test():
    test_kubernetes_pod_operator = KubernetesPodOperator(
        task_id="test_kubernetes_pod_operator",
        name="test_kubernetes_pod_operator",
        image="ubuntu",
        cmds=["/bin/bash", "-c"],
        arguments=[
            """mkdir -p /airflow/xcom/;echo '{"Log": "Hello, World!"}' > /airflow/xcom/return.json"""
        ],
        labels={"foo": "bar"},
        do_xcom_push=True,
    )

    print_contexts = PythonOperator(
        task_id="print_contexts", python_callable=_print_contexts,
    )

    test_kubernetes_pod_operator >> print_contexts


DAG = test()
```

예제는 `KubernetesPodOperator`인 `test_kubernetes_pod_operator`가 `{"Log": "Hello, World!"}`를 XCom에 작성하고 `PythonOperator`인 `print_contexts`을 통해 아래와 같이 출력한다.

```bash
# test_kubernetes_pod_operator
...
{pod.py:524} INFO - xcom result: 
{"Log": "Hello, World!"}
...
# print_contexts
...
{logging_mixin.py:149} INFO - ====================================================================================================
{logging_mixin.py:149} INFO - result_from_kubernetes_pod_operator:	 {'Log': 'Hello, World!'}
{logging_mixin.py:149} INFO - ====================================================================================================
...
```

전략을 바꿔서 아래와 같이 구성해보겠다.

1. `KubernetesPodOperator`
   1. `/var/log/fail2ban.log` 마운트
   2. `fail2ban.log`를 다음 task에 전달 (`/airflow/xcom/return.json`)
2. `PythonOperator`
   1. 이전 단계의 결과와 적재된 결과 비교
   2. 다르다면 log 파일을 작성하고 Webhook으로 메시지 전달

먼저 `fail2ban.log`를 읽고 `/airflow/xcom/return.json`를 작성하는 Python 코드를 개발하기 전 테스트를 해보자.

```python get_files.py
import json
import os

if __name__ == "__main__":
    res = {}
    res["Abs"] = os.getcwd()
    res["Results"] = os.listdir()
    os.makedirs("/airflow/xcom", exist_ok=True)
    print(res)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(res, f)
```

```docker Dockerfile
FROM python:3.8

WORKDIR /app
COPY get_files.py .

CMD ["python", "get_files.py"]
```

```shell
$ docker build -t airflow-get-files:dev .
```

이렇게 생성된 이미지를 아래와 같은 DAG로 실행할 수 있다.

```python test2.py
import json

import airflow
import requests
from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)


def _print_contexts(**kwargs):
    print("=" * 100)
    ti = kwargs["ti"]
    result_from_kubernetes_pod_operator = ti.xcom_pull(
        task_ids="test_kubernetes_pod_operator"
    )
    print("result_from_kubernetes_pod_operator:\t", result_from_kubernetes_pod_operator)
    print("=" * 100)


@dag(
    dag_id="test2",
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval="@hourly",
    max_active_runs=1,
    catchup=False,
)
def test2():
    test_kubernetes_pod_operator = KubernetesPodOperator(
        task_id="test_kubernetes_pod_operator",
        name="test_kubernetes_pod_operator",
        image="airflow-get-files:dev",
        do_xcom_push=True,
    )

    print_contexts = PythonOperator(
        task_id="print_contexts", python_callable=_print_contexts,
    )

    test_kubernetes_pod_operator >> print_contexts


DAG = test2()
```

```bash
{logging_mixin.py:149} INFO - ====================================================================================================
{logging_mixin.py:149} INFO - result_from_kubernetes_pod_operator:	 {'Abs': '/app', 'Results': ['get_files.py']}
{logging_mixin.py:149} INFO - ====================================================================================================
```

잘 실행되는 것을 확인할 수 있다!
이제 `PV`, `PVC`를 K8s에 구성하고 `KubernetesPodOperator`에 연결해보자.

```yaml storage.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: zerohertz-airflow-log-pv
  labels:
    type: zerohertz-airflow-log
spec:
  storageClassName: airflow-storage
  capacity:
    storage: 10Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: "/var/log/fail2ban.log"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zerohertz-airflow-log-pvc
  namespace: airflow
spec:
  storageClassName: airflow-storage
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  selector:
    matchLabels:
      type: zerohertz-airflow-log
```

`kubectl apply -f storage.yaml`로 K8s에 PV와 PVC를 생성하고 확인을 위해 Docker image에서 실행될 아래와 같은 Python 코드를 준비했다.


```python get_files.py
import json
import os


def print_tree(directory, prefix=""):
    if os.path.isdir(directory):
        print(prefix + "├── " + os.path.basename(directory) + "/")
        prefix = prefix + "│   "
        for item in sorted(os.listdir(directory)):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                print_tree(path, prefix)
            else:
                print(prefix + "├── " + item)
    else:
        print(prefix + "├── " + os.path.basename(directory))


if __name__ == "__main__":
    res = {}
    res["Abs"] = os.getcwd()
    res["Results"] = os.listdir()
    os.makedirs("/airflow/xcom", exist_ok=True)
    print(res)
    print_tree("/app")
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(res, f)
```

그리고 준비된 `PV`와 `PVC`를 DAG의 `KubernetesPodOperator`가 마운트하는 방법은 아래와 같다.

```python test3.py
...
from kubernetes.client.models import V1Volume, V1VolumeMount

volume_mount = V1VolumeMount(
    name="zerohertz-airflow-log-volume", mount_path="/app/fail2ban.log", read_only=True
)

volume_config = V1Volume(
    name="zerohertz-airflow-log-volume",
    persistent_volume_claim={"claimName": "zerohertz-airflow-log-pvc"},
)
...
def test3():
    test_kubernetes_pod_operator = KubernetesPodOperator(
        task_id="test_kubernetes_pod_operator",
        name="test_kubernetes_pod_operator",
        image="airflow-get-files:dev",
        do_xcom_push=True,
        volumes=[volume_config],
        volume_mounts=[volume_mount],
    )
    ...
```

```bash
# test_kubernetes_pod_operator
...
{pod_manager.py:367} INFO - {'Abs': '/app', 'Results': ['fail2ban.log', 'get_files.py']}
{pod_manager.py:367} INFO - ├── app/
{pod_manager.py:367} INFO - │   ├── fail2ban.log
{pod_manager.py:367} INFO - │   ├── get_files.py
...
# print_contexts
...
{logging_mixin.py:149} INFO - ====================================================================================================
{logging_mixin.py:149} INFO - result_from_kubernetes_pod_operator:	 {'Abs': '/app', 'Results': ['fail2ban.log', 'get_files.py']}
{logging_mixin.py:149} INFO - ====================================================================================================
...
```

## XCom을 통한 Log 전송

마운트 된 `fail2ban.log`를 읽고, `PythonOperator`에 전달하기 위해 `get_log.py`를 개발하고 Docker image를 생성한다.

```python get_log.py
import json
import os

if __name__ == "__main__":
    res = {}
    with open("fail2ban.log", "r") as f:
        log = f.readlines()
    res["log"] = log
    os.makedirs("/airflow/xcom", exist_ok=True)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(res, f)
```

```python test4.py
...
def _print_contexts(ti):
    result_from_kubernetes_pod_operator = ti.xcom_pull(
        task_ids="test_kubernetes_pod_operator"
    )
    print("result_from_kubernetes_pod_operator:\t", result_from_kubernetes_pod_operator)
    ...
>>> {logging_mixin.py:149} INFO - result_from_kubernetes_pod_operator:	 {'log': ['2023-08-13 04:24:32,227 fail2ban.server         [127633]: INFO    ---------- ...
```

출력이 잘 되는 것을 확인할 수 있다.

## Log 변경 감지

현재 로그 (`current_log`)와 적재된 로그 (`past_log`)의 길이를 비교하여 현재가 더 크다면 현재 로그를 적재하고 Discord Webhook에 두 로그의 길이 차이만큼 로그를 전달한다.
반대의 경우 에러 메시지를 보낸다.

```python
...
def _check_jail(ti):
    ...
    current_log = ti.xcom_pull(task_ids="get_current_log")
    current_log = current_log["log"]

    try:
        with open(PAST_LOG_PATH, "r") as f:
            past_log = f.readlines()
    except:
        past_log = []

    length_past_log, length_current_log = len(past_log), len(current_log)
    if length_past_log < length_current_log:
        with open(PAST_LOG_PATH, "w") as f:
            f.writelines(current_log)
        response = _send_discord_message(DISCORD_WEBHOOK, current_log[length_past_log:])
    elif length_past_log > length_current_log:
        response = _send_discord_message(
            DISCORD_WEBHOOK, "ERROR: THE PAST LOG IS LONGER THAN CURRENT LOG"
        )
    ...
```

---

# 최종 코드!

하지만 수많은 로그를 한번에 전송하거나 아주 긴 로그를 한번에 보낼 수 없다.

```python
>>> _send_discord_message(DISCORD_WEBHOOK, "="*2000)
<Response [204]>
>>> _send_discord_message(DISCORD_WEBHOOK, "="*2001)
<Response [400]>
```

따라서 아래와 같이 1초 및 줄 간격으로 메시지를 보내는 것으로 구현했다.

```python Jail.py
import json
import time

import airflow
import requests
from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)
from kubernetes.client.models import V1Volume, V1VolumeMount

volume_mount = V1VolumeMount(
    name="zerohertz-airflow-log-volume", mount_path="/app/fail2ban.log", read_only=True
)

volume_config = V1Volume(
    name="zerohertz-airflow-log-volume",
    persistent_volume_claim={"claimName": "zerohertz-airflow-log-pvc"},
)


def _send_discord_message(webhook_url, content):
    data = {"content": content}
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, data=json.dumps(data), headers=headers)
    return response


def _check_jail(ti):
    DISCORD_WEBHOOK = "${DISCORD_WEBHOOK}"
    PAST_LOG_PATH = "logs/fail2ban.log"

    current_log = ti.xcom_pull(task_ids="get_current_log")
    current_log = current_log["log"]

    try:
        with open(PAST_LOG_PATH, "r") as f:
            past_log = f.readlines()
    except:
        past_log = []

    length_past_log, length_current_log = len(past_log), len(current_log)
    if length_past_log < length_current_log:
        with open(PAST_LOG_PATH, "w") as f:
            f.writelines(current_log)
        for cl in current_log[length_past_log:]:
            response = _send_discord_message(DISCORD_WEBHOOK, "```\n" + cl + "```")
            if not response.status_code == 204:
                _send_discord_message(
                    DISCORD_WEBHOOK,
                    f"DISCORD WEBHOOK ERROR\n\tRESPONSE: {response.status_code}",
                )
                raise Exception(
                    f"DISCORD WEBHOOK ERROR\n\tRESPONSE: {response.status_code}"
                )
            time.sleep(1)
    elif length_past_log > length_current_log:
        response = _send_discord_message(
            DISCORD_WEBHOOK, "ERROR: THE PAST LOG IS LONGER THAN CURRENT LOG"
        )
        if not response.status_code == 204:
            _send_discord_message(
                DISCORD_WEBHOOK,
                f"DISCORD WEBHOOK ERROR\n\tRESPONSE: {response.status_code}",
            )
            raise Exception(
                f"DISCORD WEBHOOK ERROR\n\tRESPONSE: {response.status_code}"
            )


@dag(
    dag_id="Check-Jail",
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval="@hourly",
    max_active_runs=1,
    catchup=False,
)
def jail():
    get_current_log = KubernetesPodOperator(
        task_id="get_current_log",
        name="get_current_log",
        image="airflow-get-current-log:v1.0",
        do_xcom_push=True,
        volumes=[volume_config],
        volume_mounts=[volume_mount],
    )

    check_jail = PythonOperator(task_id="check_jail", python_callable=_check_jail,)

    get_current_log >> check_jail


DAG = jail()
```

![jail](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/260419362-6244ab72-e0c9-4366-acb3-bf5fd5022d7a.gif)

다 하고나니 아래와 같은 방법이 있었다...

---

# 허무한 결말 ~

+ [Adding ban/unban notifications from Fail2Ban to Discord!](https://technicalramblings.com/blog/adding-ban-unban-notifications-from-fail2ban-to-discord)
+ [Logging Fail2Ban to Discord](https://blog.alexsguardian.net/posts/2022/08/09/Fail2Ban+Discord/)

```bash /etc/fail2ban/jail.local (Discord)
[DEFAULT]
findtime = 1d
maxretry = 5
bantime  = 1w
backend  = systemd
ignoreip = 127.0.0.1/8 192.168.0.0/24

[sshd]
enabled = true
action  = discord_notifications
         iptables-allports
port    = ${SSH_PORT}
logpath = %(sshd_log)s
backend = %(sshd_backend)s
```

```bash /etc/fail2ban/action.d/discord_notifications.conf (Discord)
[Definition]
actionstart = curl -X POST "<webhook>" \
            -H "Content-Type: application/json" \
            -d '{"username": "Fail2Ban", "content":":white_check_mark: <hostname> - **[<name>]** jail has started"}'
actionstop = curl -X POST "<webhook>" \
            -H "Content-Type: application/json" \
            -d '{"username": "Fail2Ban", "content":":no_entry: <hostname> - **[<name>]** jail has been stopped"}'
actionban = curl -X POST "<webhook>" \
            -H "Content-Type: application/json" \
            -d '{"username":"Fail2Ban", "content":":bell: <hostname> - **[<name>]** **BANNED** IP: `<ip>` after **<failures>** failure(s). Here is some info about the IP: https://db-ip.com/<ip>. Unban by running: `fail2ban-client unban <ip>`"}'
actionunban = curl -X POST "<webhook>" \
            -H "Content-Type: application/json" \
            -d '{"username":"Fail2Ban", "content":":bell: <hostname> - **[<name>]** **UNBANNED** IP: [<ip>](https://db-ip.com/<ip>)"}'

[Init]
name = default
webhook = ${WEBHOOK}
hostname = ${HOSTNAME}
```

![fail2ban-discord](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/260435859-84884c61-463b-4bc1-a34f-ded9a2899f83.png)

심지어 대략적인 위치 정보도 제공하는 사이트를 링크해준다...
현재는 Slack으로 이전하여 사용 중 이다.

```bash /etc/fail2ban/jail.local (Slack)
[DEFAULT]
findtime = 1d
maxretry = 5
bantime  = 1w
backend  = systemd
ignoreip = 127.0.0.1/8 192.168.0.0/24

[sshd]
enabled = true
action  = slack_notifications
         iptables-allports
port    = ${SSH_PORT}
logpath = %(sshd_log)s
backend = %(sshd_backend)s
```

```bash /etc/fail2ban/action.d/slack_notifications.conf (Slack)
[Definition]
actionstart = curl -X POST https://slack.com/api/chat.postMessage \
            -H "Authorization: Bearer <token>" \
            -H "Content-type: application/json" \
            -d '{
                "channel": "zerohertz",
                "text": ":white_check_mark: <hostname> - **[<name>]** jail has started",
                "username": "Fail2Ban",
                "icon_emoji": ":bank:",
            }'
actionstop = curl -X POST https://slack.com/api/chat.postMessage \
            -H "Authorization: Bearer <token>" \
            -H "Content-type: application/json" \
            -d '{
                "channel": "zerohertz",
                "text": ":no_entry: <hostname> - **[<name>]** jail has been stopped",
                "username": "Fail2Ban",
                "icon_emoji": ":bank:",
            }'
actionban = curl -X POST https://slack.com/api/chat.postMessage \
            -H "Authorization: Bearer <token>" \
            -H "Content-type: application/json" \
            -d '{
                "channel": "zerohertz",
                "text": ":bell: <hostname> - **[<name>]** **BANNED** IP: `<ip>` after **<failures>** failure(s). Here is some info about the IP: https://db-ip.com/<ip>. Unban by running: `fail2ban-client unban <ip>`",
                "username": "Fail2Ban",
                "icon_emoji": ":bank:",
            }'
actionunban = curl -X POST https://slack.com/api/chat.postMessage \
            -H "Authorization: Bearer <token>" \
            -H "Content-type: application/json" \
            -d '{
                "channel": "zerohertz",
                "text": ":bell: <hostname> - **[<name>]** **UNBANNED** IP: [<ip>](https://db-ip.com/<ip>)",
                "username": "Fail2Ban",
                "icon_emoji": ":bank:",
            }'

[Init]
name = default
token = ${SLACK_BOT_TOKEN}
hostname = Zerohertz
```
