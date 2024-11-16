---
title: Web Crawler for NLP (1)
date: 2018-08-31 14:32:15
categories:
- Etc.
tags:
- Python
- Selenium
---
# 데이터 가공
~~~Python
from selenium import webdriver
import os

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver')  #Directory지정
browser.implicitly_wait(5)  #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=94YwFIJ-yR0&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3&index=1"
browser.get(url)

for a in range(1, 2): #url의 개수
        b = str(a)
        os.mkdir(b + "번째 기사")
        os.chdir(b + "번째 기사")
        print(a, "번째 url open")
        products = browser.find_elements_by_css_selector('#description > yt-formatted-string')
        f = open("기사 대본.txt", 'w')  #txt적기
        for product in products:
                Z = product.text
        f.write(Z[:-117])
        browser.save_screenshot("Website.png")
        os.chdir("..")
browser.quit()
~~~
<!-- more -->

![실행결과](/images/selenium-nlp-1/45676988-0d601f00-bb6e-11e8-99dd-961ac6a4fd20.png)
`Directory`만들고 거기에 `txt`파일로 내용 저장하기 성공
하지만 다음 영상을 가지고 오는 방법을 만들어야한다
~~~Python
from selenium import webdriver
import os
import time

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver')  #Directory지정
browser.implicitly_wait(5)  #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=94YwFIJ-yR0&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3&index=1"
browser.get(url)

for a in range(1, 3): #url의 개수
        b = str(a)
        os.mkdir(b + "번째 기사")
        os.chdir(b + "번째 기사")
        print(a, "번째 url open")
        products = browser.find_elements_by_css_selector('#description > yt-formatted-string')
        f = open("기사 대본.txt", 'w')  #txt적기
        for product in products:
                Z = product.text
        f.write(Z[:-117])
        browser.save_screenshot("Website.png")
        browser.find_element_by_css_selector('#movie_player > div.ytp-chrome-bottom > div.ytp-chrome-controls > div.ytp-left-controls > a.ytp-next-button.ytp-button').click()
        os.chdir("..")
        time.sleep(3)
browser.quit()
~~~
`CSS 선택자`를 통해 `Click()`함수로 해결
***
# 음성 추출
~~~Python
from selenium import webdriver
import os
import time
import pytube

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver')  #Directory지정
browser.implicitly_wait(5)  #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=94YwFIJ-yR0&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3&index=1"
browser.get(url)

for a in range(1, 3): #url의 개수
        b = str(a)
        os.mkdir(b + "번째 기사")
        os.chdir(b + "번째 기사")
        print(a, "번째 url open")
        products = browser.find_elements_by_css_selector('#description > yt-formatted-string')
        f = open("기사 대본.txt", 'w')  #txt적기
        for product in products:
                Z = product.text
        f.write(Z[:-117])
        yt = browser.current_url
        yt = pytube.YouTube(yt)
        stream = yt.streams.first()
        stream.download()
        browser.save_screenshot("Website.png")
        browser.find_element_by_css_selector('#movie_player > div.ytp-chrome-bottom > div.ytp-chrome-controls > div.ytp-left-controls > a.ytp-next-button.ytp-button').click()
        os.chdir("..")
        time.sleep(3)
browser.quit()
~~~
![실행결과](/images/selenium-nlp-1/45679349-6f6f5300-bb73-11e8-9bea-a706e2730898.png)
동영상 추출 성공!
