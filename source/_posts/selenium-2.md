---
title: Web Crawling by Selenium (2)
date: 2018-08-30 22:58:37
categories:
- Etc.
tags:
- Python
- Selenium
- BeautifulSoup
---
`Docker`로 부서진 멘탈을 복구시키고...
# Selenium의 스크레이핑
> Selenium을 import하기

~~~Python
from selenium import Webdriver
~~~
> 대응되는 Driver

~~~Python
Webdriver.Firefox
Webdriver.Chrome
Webdriver.Ie
Webdriver.Opera
Webdriver.PhantomJS
Webdriver.Remote
~~~
> Selenium으로 DOM 요소 선택

`74쪽`

<!-- more -->

***
# 대략적인 Design
~~~
Make directory && Change directory
PhantomJS로 네이버 뉴스 접속
동영상을 mp3 파일로 저장
BeautifulSoup로 대사 저장
cd ..
무한 반복
~~~
`mp3 저장`, `대사 데이터 가공` 제외 구현
~~~Python
from selenium import webdriver
from bs4 import BeautifulSoup
import os

browser = webdriver.PhantomJS('/Users/OHG/phantomjs-2.1.1-windows/bin/phantomjs')  #Directory지정
browser.implicitly_wait(3)  #암묵적으로 3초 딜레이

for a in range(50, 72): #url의 개수
    url = "https://news.naver.com/main/read.nhn?mode=LPOD&mid=tvh&oid=055&aid=00006709" + str(a)
    browser.get(url)
    b = str(a)
    os.mkdir(b + "번째 기사")
    os.chdir(b + "번째 기사")
    print(a, "번째 url open")
    html = browser.page_source  #페이지의 elements 모두 가져오기
    soup = BeautifulSoup(html, 'html.parser')  #BeautifulSoup사용하기
    products = soup.select('body > div > table > tbody > tr > td > div > div br')
    products = browser.find_element_by_id('main_content')
    f = open("기사 대본.txt", 'w') #txt적기
    f.write(products.text)
    os.chdir("..")
~~~
하지만 `url`이 들어갈때 동영상만 있는 기사를 고르려면 `PhantomJS`로 들어가서 `url`을 다시 따와야 한다.
~~~Python
from selenium import webdriver
from bs4 import BeautifulSoup
import os

browser = webdriver.PhantomJS('/Users/OHG/phantomjs-2.1.1-windows/bin/phantomjs')  #Directory지정
browser.implicitly_wait(3)  #암묵적으로 3초 딜레이

for a in range(1,8):
    html1 = "https://news.naver.com/main/tv/list.nhn?mode=LPOD&mid=tvh&oid=055&date=20180830&page=" + str(a)
    soup = BeautifulSoup(html1, 'html.parser')
    links = soup.find_all("div > table > tbody > tr > td.content > div > div > ul > li > dl > dt a")
    for l in links:
        name = l.string
        print(name)
    for b in range(70, 72):  #url의 개수
        url = "https://news.naver.com/main/read.nhn?mode=LPOD&mid=tvh&oid=055&aid=00006709" + str(b)
        browser.get(url)
        c = str(b)
        os.mkdir(c + "번째 기사")
        os.chdir(c + "번째 기사")
        print(b, "번째 url open")
        html2 = browser.page_source  #페이지의 elements 모두 가져오기
        soup = BeautifulSoup(html2, 'html.parser')  #BeautifulSoup사용하기
        products = soup.select('body > div > table > tbody > tr > td > div > div br')
        products = browser.find_element_by_id('main_content')
        f = open("기사 대본.txt", 'w')  #txt적기
        f.write(products.text)
        os.chdir("..")
~~~
생각보다 어려우니 `BeautifulSoup`를 이용해서 `<a>` 태그만 따오는걸 만들자
원인을 찾았다... `HTTPS`통신을 해서 `BeautifulSoup`가 적용이 안됐다.`Selenium`을 이용해서 크롤링해보자.
~~~Python
from selenium import webdriver
from bs4 import BeautifulSoup
import os

browser = webdriver.PhantomJS('/Users/OHG/phantomjs-2.1.1-windows/bin/phantomjs')  #Directory지정
browser.implicitly_wait(3)  #암묵적으로 3초 딜레이

for a in range(10, 15):  # url의 개수
        url = "https://news.naver.com/main/read.nhn?mode=LPOD&mid=tvh&oid=437&aid=00001903" + str(b)
        browser.get(url)
        b = str(a)
        os.mkdir(b + "번째 기사")
        os.chdir(b + "번째 기사")
        print(a, "번째 url open")
        html2 = browser.page_source  #페이지의 elements 모두 가져오기
        soup = BeautifulSoup(html2, 'html.parser')  #BeautifulSoup사용하기
        products = soup.select('body > div > table > tbody > tr > td > div > div br')
        products = browser.find_element_by_id('main_content')
        f = open("기사 대본.txt", 'w')  #txt적기
        f.write(products.text)
        os.chdir("..")
~~~
갓갓 JTBC는 그냥 다 동영상이다
![실행결과](/images/selenium-2/44866181-7b2bd000-acbf-11e8-8ef2-7ba85d004656.png)
![실행결과](/images/selenium-2/44866239-a44c6080-acbf-11e8-98d0-54bb11b3b689.png)
***
# 앞으로 보완해야할 것
+ 데이터 가공
+ mp3 음성 추출
