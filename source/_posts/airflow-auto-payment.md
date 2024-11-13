---
title: Apache Airflow 기반 자동결제 DAG 개발
date: 2023-08-18 15:26:28
categories:
- 3. DevOps
tags:
- Airflow
- Python
- Docker
- Kubernetes
- Selenium
- Home Server
---
# Introduction

신용카드 중 [The More](https://www.shinhancard.com/pconts/html/card/apply/credit/1198942_2207.html)는 5,000원 이상의 결제 금액에 대해 1,000원 미만의 포인트가 적립된다.
이를 통신비와 같은 서비스에 분할결제로 적용하면 약 16.65%의 이득을 볼 수 있다. ($\because\frac{999}{5999}\times100$)
하지만 아래와 같은 제약이 존재한다.

> 동일한 가맹점의 경우 1일 1회에 한하여 포인트 적립이 되며, ...

따라서 하루에 한 번만 The More 카드로 5,999원이 결제되도록 Apache Airflow의 DAG를 구성한다.
물론 [논란](https://namu.wiki/w/%EC%8B%A0%ED%95%9C%EC%B9%B4%EB%93%9C%20%EB%B6%84%ED%95%A0%EA%B2%B0%EC%A0%9C%20%EC%A0%9C%ED%95%9C%20%EC%82%AC%EA%B1%B4)은 많지만 개발도 연습하기 좋은 예제였다.

<!-- More -->

---

# DAG

결제를 하려면 자신이 사용하는 통신사의 Web을 접속해야하기 때문에 [Selenium](https://www.selenium.dev/)을 사용해 로그인, 결제 정보 입력을 진행했다.
하지만 Airflow 환경에서 여러 dependency를 설치하고 관리하기 어렵기 때문에 `KubernetesPodOperator`를 사용해서 모든 task를 진행하는 Docker image를 생성하고 아래와 같이 실행되게 했다.

```python
import airflow
from airflow.decorators import dag
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)

@dag(
    dag_id="Uplus",
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval="0 0 * * *",
    max_active_runs=1,
    catchup=False,
)
def Uplus():
    Uplus = KubernetesPodOperator(
        task_id="Uplus",
        name="Uplus",
        image="airflow-uplus:v1",
    )

    Uplus


DAG = Uplus()
```

아침 9시에 한 번씩 준비된 Docker image `airflow-uplus:v1`을 실행하게 된다.
해당 이미지를 생성하기 위해 아래의 코드를 사용했다.

---

# Docker Image

Selenium을 통해 U+ web에 접속하고 결제 정보 입력 및 진행에 대한 코드는 아래와 같다.
마지막 자동결제 건은 2배 적립이기 때문에 조건문을 구성했다.
또한 결제가 잘되는지, 결제가 진행됐는지 여부를 파악하기 위해 discord webhook을 사용했다.

```python Uplus.py
import json
import time

import requests
from selenium import webdriver
from selenium.webdriver.support.ui import Select

# Login ID
USER_ID =
# Login Password
USER_PASSWORD =
# 결제에 사용할 카드 번호
CARD_NO =
# U+ 사용자 이름
NAME =
# 생년월일
BIRTH =
# 카드 만료 년도
CARD_YEAR =
# 카드 만료 월
CARD_MONTH =
# DISCORD WEBHOOK
WEBHOOK =


def xpath_click(browser, element):
    element = browser.find_element("xpath", element)
    element.click()


def id_send(browser, element, key):
    element = browser.find_element("id", element)
    element.send_keys(key)


def name_send(browser, element, key):
    element = browser.find_element("name", element)
    element.send_keys(key)


def id_select(browser, element, key):
    element = browser.find_element("id", element)
    select = Select(element)
    select.select_by_value(key)


def get_price(browser):
    element = browser.find_element(
        "xpath",
        "/html/body/div[6]/div[1]/div/div/div/div/div[1]/div/div/div[1]/div/p/strong",
    )
    return int(element.text[:-1].replace(",", ""))


def price_send(browser, PRICE):
    element = browser.find_element("id", "displayPayAmt")
    browser.execute_script("arguments[0].value = '';", element)
    element = browser.find_element("xpath", '//*[@id="displayPayAmt"]')
    element.send_keys(PRICE)


def login(browser):
    id_send(browser, "username-1-6", USER_ID)
    id_send(browser, "password-1", USER_PASSWORD)
    xpath_click(
        browser,
        "/html/body/div[1]/div/div/div[4]/div[1]/div/div[2]/div/div/div/div/section/div/button",
    )
    xpath_click(
        browser,
        "/html/body/div[1]/div/div/div[4]/div[1]/div/div[2]/div/div/div/div/section/div/button",
    )
    time.sleep(5)


def move(browser):
    browser.get("https://www.lguplus.com/mypage/payinfo?p=1")
    time.sleep(3)
    xpath_click(
        browser,
        "/html/body/div[1]/div/div/div[4]/div[1]/div/div[2]/div/div/div/div[2]/div[1]/div/div[3]/button[1]",
    )
    time.sleep(8)


def info(browser, PRICE):
    id_send(browser, "cardNo", CARD_NO)
    name_send(browser, "cardCustName", NAME)
    name_send(browser, "cardCustbirth", BIRTH)
    id_select(browser, "selCardDate1", CARD_YEAR)
    id_select(browser, "selCardDate2", CARD_MONTH)
    price_send(browser, PRICE)
    price_send(browser, PRICE)


def send_discord_message(webhook_url, content):
    data = {"content": content}
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, data=json.dumps(data), headers=headers)
    return response


if __name__ == "__main__":
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        )
        browser = webdriver.Chrome(options)

        # U+ 접속
        browser.get("https://www.lguplus.com/login/onid-login")

        # U+ 로그인
        login(browser)

        # 결제 화면 이동
        move(browser)

        # 결제 잔액 확인
        tmp = get_price(browser)

        # 결제 가격
        if tmp == 0 or tmp == 5999:
            send_discord_message(WEBHOOK, f":no_bell: [결제 :x:] 자동결제 금액:\t{tmp}원")
            exit()
        elif tmp > 5999 + 5999:
            PRICE = "5999"
        else:
            PRICE = str(tmp - 5999)

        # 결제 정보 입력
        send_discord_message(WEBHOOK, f":bell: [결제 :o:] 결제 예정 금액:\t{PRICE}원")
        info(browser, PRICE)

        # 결제
        xpath_click(browser, "/html/body/div[6]/div[1]/div/div/footer/button[2]")
        send_discord_message(WEBHOOK, f":bell: [결제 :o:] 결제 완료!:\t{PRICE}원")
        send_discord_message(
            WEBHOOK, f":bell: [결제 :o:] 결제 후 결제 예정 금액:\t{tmp - int(PRICE)}원"
        )
    except Exception as e:
        send_discord_message(
            WEBHOOK,
            ":warning:" * 10
            + "ERROR!!!"
            + ":warning:" * 10
            + "\n"
            + "```\n"
            + str(e)
            + "\n```",
        )
```

마지막으로 이 코드를 `selenium/standalone-chrome` image와 함께 구성하고 `docker build -t airflow-uplus:v1 .`로 빌드하면 끝이다.

```docker Dockerfile
FROM selenium/standalone-chrome

USER root
RUN apt-get update \
    && apt-get install -y python3 python3-pip

RUN pip install requests

COPY Uplus.py /app/Uplus.py

WORKDIR /app

RUN pip3 install selenium

CMD ["python3", "Uplus.py"]
```

이제 매일 결제할 필요가 없어졌다.

<img width="381" alt="Result" src="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/261619194-e0d1710a-852a-438d-9839-f63dcfa1eceb.png">