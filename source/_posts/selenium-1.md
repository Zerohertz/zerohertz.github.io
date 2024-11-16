---
title: Web Crawling by Selenium (1)
date: 2018-08-30 10:36:35
categories:
- Etc.
tags:
- Python
- Selenium
- BeautifulSoup
---
# Selenium
> `Javascript`를 많이 사용하는 웹 사이트는 웹 브라우저를 사용하지 않으면 제대로 동작하지 않아 `Requsest` 모듈로 대처할 수 없다.

그래서 웹 브라우저를 원격을 조작할 수 있는 도구 `Selenium`을 사용한다.
+ 자동으로 URL을 열고 클릭할 수 있다
+ 스크롤하거나, 문자를 입력할 수 있다
+ 화면을 캡처해서 이미지로 저장하거나 `HTML`의 특정 부분을 꺼내는 것도 가능하다
+ 여러 다양한 조작을 자동화할 수 있다
+ 다양한 웹 브라우저에 대응한다
  
<!-- more -->
***
# PhantomJS
> 화면 없이 명령줄에서 사용할 수 있는 웹 브라우저

# Docker
> LXC(LinuX Container)와 Docker Union 파일 시스템을 사용해 변경 내용을 관리

~~~
C:\Users\OHG>docker pull ubuntu:16.04
C:\Users\OHG>docker run -it ubuntu:16.04
~~~
`Docker`에 `Ubuntu` 이미지를 가져온 후 실행하여 셸에 들어간다
~~~
root@fb769ed27a3d:/# apt-get update
root@fb769ed27a3d:/# apt-get install -y python3 python3-pip
root@fb769ed27a3d:/# pip3 install selenium
root@fb769ed27a3d:/# pip3 install beautifulsoup4
~~~
`Python3`, `pip3`, `Selenium`, `Beautifulsoup4` 설치
~~~
root@fb769ed27a3d:/# apt-get install -y wget libfontconfig
root@fb769ed27a3d:/# mkdir -p /home/root/src && cd $_
root@fb769ed27a3d:/home/root/src# wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2
root@fb769ed27a3d:/home/root/src# tar jxvf phantomjs-2.1.1-linux-x86_64.tar.bz2
root@fb769ed27a3d:/home/root/src# cd phantomjs-2.1.1-linux-x86_64/bin/
root@fb769ed27a3d:/home/root/src/phantomjs-2.1.1-linux-x86_64/bin# cp phantomjs /usr/local/bin/
~~~
`PhantomJS`에 필요한 라이브러리 설치, 바이너리 내려받고 설치
`root@fb769ed27a3d:/home/root/src/phantomjs-2.1.1-linux-x86_64/bin# apt-get install -y fonts-nanum*` : 한글 폰트 설치
~~~
root@fb769ed27a3d:/home/root/src/phantomjs-2.1.1-linux-x86_64/bin# exit
exit
~~~
~~~
C:\Users\OHG>docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                      PORTS               NAMES
fb769ed27a3d        ubuntu:16.04        "/bin/bash"         22 minutes ago      Exited (0) 25 seconds ago                       reverent_villani

C:\Users\OHG>docker commit fb769ed27a3d ubuntu-phantomjs
sha256:f62e3dc792ad314a53213cab84a311b51c1fdf9135e835dc58095f066fe95c4d
~~~
`Docker`를 사용하다가 멘탈이 나가서 버리기로 했다
***
# PhantomJS와 Selenium으로 이미지 캡쳐
~~~Python
from selenium import webdriver

url = "http://www.naver.com/"

browser = webdriver.PhantomJS('/Users/OHG/phantomjs-2.1.1-windows/bin/phantomjs') #Directory지정
browser.implicitly_wait(3) #암묵적으로 3초 딜레이
browser.get(url)
browser.save_screenshot("Website.png") #저장
browser.quit()
~~~
![results](/images/selenium-1/results.png)
`Docker`로 삽질을 어마무시하게 했다...
# 네이버에 로그인해서 구매한 물건 목록 가져오기
~~~Python
from selenium import webdriver
from bs4 import BeautifulSoup

USER = "아이디"
PASS = "비밀번호"

url = "http://nid.naver.com/nidlogin.login"

browser = webdriver.PhantomJS('/Users/OHG/phantomjs-2.1.1-windows/bin/phantomjs') #Directory지정
browser.implicitly_wait(3) #암묵적으로 3초 딜레이

browser.get(url)
print("로그인 페이지로 접근")

browser.find_element_by_name('id').send_keys(USER)
browser.find_element_by_name('pw').send_keys(PASS)

form = browser.find_element_by_css_selector("input.btn_global[type=submit]")
form.submit()
print("로그인 버튼을 누름")

browser.get("https://order.pay.naver.com/home?tabMenu=SHOPPING")

html = browser.page_source #페이지의 elements 모두 가져오기
soup = BeautifulSoup(html, 'html.parser') #BeautifulSoup사용하기
products = soup.select('div.p_inr > div.p_info > a span')

for product in products:
    print("-", product.string)
~~~
> 실행결과

~~~
C:\Users\OHG\AppData\Local\Programs\Python\Python37\python.exe "C:/Users/OHG/PycharmProjects/네이버 로그인/Main.py"
C:\Users\OHG\AppData\Local\Programs\Python\Python37\lib\site-packages\selenium\webdriver\phantomjs\webdriver.py:49: UserWarning: Selenium support for PhantomJS has been deprecated, please use headless versions of Chrome or Firefox instead
  warnings.warn('Selenium support for PhantomJS has been deprecated, please use headless '
로그인 페이지로 접근
로그인 버튼을 누름
- 
							[saiskai] 아이폰 슬림 하드 케이스
						
- 
							0.3mm 카메라 렌즈 보호 필름 
						
- None
- None
- 
							[Volkswagen] 폭스바겐 VW1425 시리즈 본사정품 남성용
						
- 
							[힐링쉴드] LG V20 AG Nanovid 저반사 지문방지 액정보호필름 2매 (HS164741)
						
- 
							LG V20 케이스
						

Process finished with exit code 0
~~~
약간 이상하게 나오는데 `BeautifulSoup`의 공부가 더 필요하다
