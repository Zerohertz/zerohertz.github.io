---
title: Web Crawling by BeautifulSoup (4)
date: 2018-08-22 16:46:04
categories:
- Etc.
tags:
- Python
- BeautifulSoup
---
# HTTP 통신
> 웹 브라우저와 웹 서버는 HTTP라는 통신규약(`프로토콜`)을 사용해서 통신한다.

브라우저에서 서버로 요청(`request`), 서버에서 브라우저로 응답(`response`)할 때 어떻게 할지를 나타낸 규약이다.
***
# 쿠키
무상태(`stateless`) `HTTP 통신`으로는 회원제 사이트를 만들 수 없다.(`stateless` : 이전에 어떤 데이터를 가져갔는지 등에 대한 정보(상태 : `state`)를 전혀 저장하지 않는 통신)
> 방문하는 사람의 컴퓨터에 일시적으로 데이터를 저장하는 기능

하지만 1개의 `쿠키`엔 `4096byte`의 데이터 크기 제한이 있다.
또한 `쿠키`는 `HTTP 통신` 헤더를 통해 읽고 쓸 수 있다. 따라서 방문자 혹은 확인자가 원하는 대로 변경할 수 있다.
하지만 위의 말대로 쉽게 변경이 가능하기에 비밀번호 등의 비밀 정보는 `세션`을 통해 다뤄진다.
`새션`도 `쿠키`를 사용해 데이터를 저장하는 점은 같다. 하지만, 방문자 고유 ID만을 저장하고, 실제로 모든 데이터는 웹 서버에 저장하므로 `쿠키`와는 다르게 저장할 수 있는 데이터에 제한이 없다.
<!-- more -->
***
# Requests
`urllib.request`로 쿠키를 이용한 접근이 가능하다.
하지만 방법이 복잡하니 `requests`라는 패키지를 사용한다.
~~~
$ pip3 install requsts
~~~
***
# Requests 사용
![](/images/beautifulsoup-4/44457939-9d7a7980-a63f-11e8-8458-29c960ce7ebd.png)
~~~Python
import requests
from bs4 import BeautifulSoup

USER = "아이디" #아이디, 비밀번호 지정
PASS = "비밀번호"

session = requests.session() #세션 시작

login_info = { #아이디, 비밀번호 지정
    "m_id": USER,
    "m_passwd": PASS
}
url_login = "http://www.hanbit.co.kr/member/login_proc.php"
res = session.post(url_login, data=login_info)
res.raise_for_status() #오류 -> 예외

url_mypage = "http://www.hanbit.co.kr/myhanbit/myhanbit.html" #마이페이지 접속
res = session.get(url_mypage)
res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser") #마일리지 크롤링
name = soup.select_one(".mileage_section1 span").get_text()
print("마일리지", name)
~~~
![실행결과](/images/beautifulsoup-4/44457941-9eaba680-a63f-11e8-91c4-58b950efd766.png)
