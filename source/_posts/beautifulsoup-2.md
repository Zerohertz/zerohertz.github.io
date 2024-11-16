---
title: Web Crawling by BeautifulSoup (2)
date: 2018-08-19 16:33:02
categories:
- Etc.
tags:
- Python
- BeautifulSoup
---
# DOM 요소의 속성 추출
```Python
C:\Users\OHG>python
Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:59:51) [MSC v.1914 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from bs4 import BeautifulSoup
>>> soup=BeautifulSoup(
...     "<p><a href='a.html'>test</a></p>",
...     "html.parser")
>>> soup.prettify()
'<p>\n <a href="a.html">\n  test\n </a>\n</p>'
>>> a=soup.p.a
>>> type(a.attrs)
<class 'dict'>
>>> 'href' in a.attrs
True
>>> a['href']
'a.html'
```
<!-- more -->
`cmd`에서 `prettify()` 메서드를 이용하여 제대로 분석이 됐는지 확인했다.
또한 `attrs` 속성의 자료형은 딕셔너리(`dict`)임을 알 수 있다.
***
# urlopen()과 BeautifulSoup조합하기
~~~Python
from bs4 import BeautifulSoup
import urllib.request as req

url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

res = req.urlopen(url) #urlopen으로 url열기

soup = BeautifulSoup(res, "html.parser") #BeautifulSoup로 분석하기

title = soup.find("title").string #태그 추출, 결과 출력
wf = soup.find("wf").string
print(title)
print(wf)
~~~
기상청 `RSS`에서 XML데이터를 추출하여 XML의 내용을 출력한다.
> 실행결과

~~~Python
C:\Users\OHG\PycharmProjects\Webcrawling2>python main.py
기상청 육상 중기예보
이번 예보기간에는 북태평양고기압의 영향을 주로 받는 가운데, 가끔 구름이 많겠습니다.<br />기온은 평년(최저기온: 18~23℃, 최고기온: 26~31℃)보다 높겠습니다.<br />강수량은 평년(6~17mm)보다 적겠습니다.<br /><br />* 이번 예보기간에도 무더위가 계속 이어지겠고, 밤에는 열대야가 나타나는 곳이 있겠습니다.<br />* 주의보 수준의 폭염이 당분간 이어짐에 따라 온열질환자 발 생 가능성이 있으니, 건강관리와 농.축산물과 수산물 관리에 유의하기 바랍니다.<br />* 제19호 태풍 '솔릭(SOULIK)'이 북상하는 가운데, 이 태풍의 발달과 이동경로에 따라 전반에 기압계 변동 가능성이 크겠으니, 앞으로 발표되는 기상정보를 참고하기 바랍니다.
~~~
***
# 네이버 금융에서 환율 정보 추출하기
~~~Python
from bs4 import BeautifulSoup
import urllib.request as req

url = "https://finance.naver.com/marketindex/" #HTML 가져오기
res = req.urlopen(url)

soup = BeautifulSoup(res, "html.parser") #HTML 분석

price = soup.select_one("div.head_info > span.value").string #데이터 추출
print("usd/krw =", price)
~~~
![HTML분석](/images/beautifulsoup-2/44307044-740ae500-a3d6-11e8-81ce-58d78041d9ef.png)
위에 있는 소스가 환율 정보다.
> 실행결과

~~~Python
C:\Users\OHG\PycharmProjects\Webcrawling3\venv\Scripts\python.exe C:/Users/OHG/PycharmProjects/Webcrawling3/Main.py
usd/krw = 1,125.00

Process finished with exit code 0
~~~
