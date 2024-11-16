---
title: Web Crawler for NLP (3)
date: 2018-09-20 15:01:38
categories:
- Etc.
tags:
- Python
- Selenium
---
# YouTube에서 JTBC로 가기
> 제목을 따와서 JTBC에 검색하자

~~~Python
from selenium import webdriver
import time

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #Directory지정
browser.implicitly_wait(5) #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=C7qr7-NlNw8&index=1&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3"
ser = "http://jtbc.joins.com/search?term="

browser.get(url)
for a in range(1, 100): #url의 개수
    products = browser.find_elements_by_css_selector('#container > h1 > yt-formatted-string')
    for product in products:
        Z = product.text
    print(Z[13:])
    time.sleep(3)
browser.quit()
~~~
> 실행결과

~~~Python
'One sweet dream'
~~~
제목 따오고 `ser`에 붙여서 검색하자
<!-- more -->
~~~Python
from selenium import webdriver
import time

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #Directory지정
driver = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver')
browser.implicitly_wait(5) #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=C7qr7-NlNw8&index=1&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3"
ser = "http://jtbc.joins.com/search?term="

browser.get(url)
for a in range(1, 100): #url의 개수
    products = browser.find_elements_by_css_selector('#container > h1 > yt-formatted-string')
    for product in products:
        Z = product.text
    S = Z[13:]
    driver.get(ser+S)
    time.sleep(3)
browser.quit()
~~~

# JTBC에서 대사 따오기
> 기사 원문 들어가기

~~~Python
from selenium import webdriver
import time

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #Directory지정, 유튜브
driver = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #검색
browser.implicitly_wait(5) #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=C7qr7-NlNw8&index=1&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3" #유튜브
ser = "http://jtbc.joins.com/search?term=" #검색

browser.get(url)
for a in range(1, 100): #url의 개수
    products = browser.find_elements_by_css_selector('#container > h1 > yt-formatted-string')
    for product in products:
        Z = product.text
    S = Z[13:]
    driver.get(ser+S)# content > div.wrap_result.clfix > div.wrap_sch_area > div.area_sch_section.last > div > div > ul > li > div > a
    driver.find_element_by_css_selector('#content > div.wrap_result.clfix > div.wrap_sch_area > div.area_sch_section.last > div > div > ul > li > div > a').click()
    time.sleep(3)
browser.quit()
~~~
> 최종 기사 따오기

~~~Python
from selenium import webdriver
import time

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #Directory지정, 유튜브
driver = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #검색
browser.implicitly_wait(5) #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=C7qr7-NlNw8&index=1&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3" #유튜브
ser = "http://jtbc.joins.com/search?term=" #검색

browser.get(url)
for a in range(1, 100): #url의 개수
    titles = browser.find_elements_by_css_selector('#container > h1 > yt-formatted-string')
    for title in titles:
        Z = title.text
    S = Z[13:]
    driver.get(ser+S)# content > div.wrap_result.clfix > div.wrap_sch_area > div.area_sch_section.last > div > div > ul > li > div > a
    driver.find_element_by_css_selector('#content > div.wrap_result.clfix > div.wrap_sch_area > div.area_sch_section.last > div > div > ul > li > div > a').click()
    abc = driver.find_elements_by_css_selector('#articlebody > div:nth-child(1)')
    for tex in abc:
        print(tex.text)
    time.sleep(3)
browser.quit()
~~~
> 실행결과

~~~
C:\Users\OHG\AppData\Local\Programs\Python\Python37\python.exe C:/Users/OHG/PycharmProjects/WCFNtest/Main.py
뉴스룸의 앵커브리핑을 시작하겠습니다.

석달 전에 북미 정상회담 중계를 위해서 싱가포르에 갔을 때 점심을 먹었던 한 식당에서 틀어놓은 노래가 유독 귀를 잡아끌었습니다.

" Is it getting better or do you feel the same
기분이 나아지고 있나요 아니면 같나요?"
- U2의 < One >

그룹 U2 의 명곡 < One > 이었습니다.

노래를 고른 식당의 주인은 무언가를 알고 있었을까.

아니면 그저 우연의 일치일까…

91년 발표된 이 곡이 녹음된 시기는 1990년 10월.

장소는 독일 베를린의 한자 스튜디오였습니다.

당시 U2의 멤버들은 해체를 고민할 정도로 갈등이 심했다고 전해지죠.

정통 록을 고집할 것인가.

실험적인 전자음을 강조할 것인가를 두고 급기야 주먹다짐까지 벌일 정도로 갈등은 심각했지만.

그곳 베를린에서는 정반대의 일이 진행되고 있었습니다.

같은 시기 견고했던 동·서독 간의 장벽은 무너져서 사람들은 미움 대신 통일을 이야기하고 있었던 것입니다.

"그러나 우리는 같은 것은 아니죠
우리는 가까이 가까이 서로에게 다가갑니다."
- U2의 < One >

U2의 명곡 < One > 은 바로 그 자리에서 30분 만에 곡조가 만들어졌고 보컬이자 리더인 보노가 가사를 붙여서 완성되었습니다.

"한국에서 가장 부르고 싶은 노래가 바로 One 이다"

그들 역시 대부분, 분단과 전쟁의 아픔을 겪은 아일랜드사람이니까요.

물론 돌파해야 할 난관은 앞으로도 많겠지만 오늘의 결과물은 달라진 서로의 관계를 실감하게 하고 있습니다.

역사적 소명이라든가 민족의 대장정 같은 무거운 단어들을 동원하지 않고서도 그저 별일 없이 서로 돕고, 그래서 함께 윤택해질 수 있는 지극히 평범한 세상.

철조망이 거둬진 철새의 땅과 다시 만나게 될 헤어진 사람들 그리고 기차를 타고 대륙으로 향하는 아득한 꿈같은 것들 말입니다.

그룹 U2의 < One > 을 듣고 나온 식당의 바로 옆 가게는 아이스크림 등을 파는 디저트 가게였는데 그 가게의 선전 문구는 공교롭게도 이랬습니다.

"Sweet dreams are made of this!
달콤한 꿈은 이것으로 만들어진다네"
- Eurythmics < Sweet  Dreams >

Eurythmics의 명곡이었습니다.

오늘의 앵커브리핑이었습니다.
~~~
***
# 마무리 코드
~~~Python
import os
from selenium import webdriver
import time
import pytube

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #Directory지정, 유튜브
driver = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver') #검색
browser.implicitly_wait(5) #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=C7qr7-NlNw8&index=1&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3" #유튜브
ser = "http://jtbc.joins.com/search?term=" #검색

browser.get(url)

for a in range(1, 100): #url의 개수
        b = str(a)
        os.mkdir(b + "번째 기사")
        os.chdir(b + "번째 기사")
        print(a, "번째 url open")
        titles = browser.find_elements_by_css_selector('#container > h1 > yt-formatted-string')
        f = open("기사 대본.txt", 'w')  #txt적기
        for title in titles:
                Z = title.text
        S = Z[13:]
        driver.get(ser + S)  # content > div.wrap_result.clfix > div.wrap_sch_area > div.area_sch_section.last > div > div > ul > li > div > a
        driver.find_element_by_css_selector('#content > div.wrap_result.clfix > div.wrap_sch_area > div.area_sch_section.last > div > div > ul > li > div > a').click()
        abc = driver.find_elements_by_css_selector('#articlebody > div:nth-child(1)')
        for tex in abc:
                f.write(tex.text)
        yt = browser.current_url
        yt = pytube.YouTube(yt)
        stream = yt.streams.all()
        stream[15].download(os.getcwd(), "내용")
        browser.save_screenshot("Website.png")
        browser.find_element_by_css_selector('#movie_player > div.ytp-chrome-bottom > div.ytp-chrome-controls > div.ytp-left-controls > a.ytp-next-button.ytp-button').click()
        os.chdir("..")
        time.sleep(3)
browser.quit()
~~~
![실행결과](/images/selenium-nlp-3/45914919-d0a26980-be86-11e8-9a79-83596d3f83bc.png)