---
title: Web Crawling by BeautifulSoup (3)
date: 2018-08-22 14:14:09
categories:
- Etc.
tags:
- Python
- BeautifulSoup
---
# 새로운 파이썬 파일에 bs4 추가하기(PyCharm)
![pycharm-1](/images/beautifulsoup-3/pycharm-1.png)
![pycharm-2](/images/beautifulsoup-3/pycharm-2.png)
`bs4`를 검색하여 찾은 후 install한다.
***
# HTML 구조 확인하기
크롬에서 `Ctrl+Shift+i`키를 눌러서 개발자 도구를 열어서 확인할 수 있다.
![element](/images/beautifulsoup-3/element.png)
개발자 도구 팝업 메뉴에서 `Copy > Copy selector`을 사용하면 선택한 요소의 CSS 선택자가 클립보드에 복사된다.
<!-- more -->
***
# Wiki에서 화제의 단어 크롤링하기
![css-selector](/images/beautifulsoup-3/css-selector.png)
CSS 선택자의 결과
`#mw-content-text > div > table:nth-child(3) > tbody > tr > td:nth-child(1) > table:nth-child(6) > tbody > tr:nth-child(2) > td > ul:nth-child(4) > li > a:nth-child(1)`
`nth-child(n)`은 n번째의 요소를 말한다.(`BeautifulSoup`는 `nth-child`를 지원하지 않는다.)
~~~Python
from bs4 import BeautifulSoup
import urllib.request as req

url = "https://ko.wikipedia.org/wiki/%ED%8F%AC%ED%84%B8:%EC%9A%94%EC%A6%98_%ED%99%94%EC%A0%9C" #url
res = req.urlopen(url)
soup = BeautifulSoup(res, "html.parser")

a_list = soup.select("#mw-content-text > div > table > tbody > tr > td > table > tbody > tr > td > ul > li a") #a 태그 전까지의 모든 태그 아래의 모든 a 태그를 모두 선택

for a in a_list:
        name = a.string
        print("-", name)
~~~
> 실행결과

~~~
C:\Users\OHG\PycharmProjects\Webcrawling4\venv\Scripts\python.exe C:/Users/OHG/PycharmProjects/Webcrawling4/Main.py
- 8월 19일
- 인도네시아
- 롬복섬
- M6.9의 지진
- 8월 18일
- 자카르타
- 팔렘방
- 아시안 게임
- 8월 14일
- 이탈리아
- 제노바
- 모란디 교
- 8월 6일
- 방글라데시
- 다카
- 대규모 시위와 충돌
- 7월 24일
- 그리스
- 아티키 주
- 산불
- 대한민국
- 국방부
- 군기교육
- 엘살바도르
- 중화민국
- 중화인민공화국
- 5월
- 도미니카 공화국
- 부르키나파소
- 대한민국
- 경상북도
- 봉화군
- 소천면
- 세일전자 화재 사고
- 인천광역시
- 남동구
- 2018년 아시안 게임
- 류샹
- 자오징
- 베네수엘라
- 모멘트 규모
- 제19호 태풍 솔릭
- 기상청
- 20일
- 제주도
- 충청남도
- 금강산
- 남북 이산가족 상봉
- 2018년 아시안 게임
- 일본 남자 농구 국가대표팀
- 자카르타
- 대한민국
- 강원지방경찰청
- 강원도
- 제19호 태풍 솔릭
- 태풍 산바
- 한반도
- 기상청
- 제주도
- 전라남도
- 대한민국
- 경기도
- 과천시
- 서울대공원
- 경찰
- 서해안고속도로
- 서산휴게소
- 피의자
- 2018년 8월 19일 롬복섬 지진
- 인도네시아
- 롬복
- 2018년 아시안 게임
- 인도네시아
- 자카르타
- 겔로라 붕 카르노 스타디움
- 유엔 사무총장
- 코피 아난
- 필리핀
- 마닐라
- 니노이 아키노 국제공항
- 활주로
- 아레사 프랭클린
- 서울북부지방법원
- 인도
- 케랄라 주
- 홍수
- 태풍 솔릭
- 대한민국
- 김경수
- 경상남도지사
- 국무회의
- 군사안보지원사령부
- 국군기무사령부
- 이탈리아
- 제노바
- 모란디 다리
- 대한민국
- BMW
- 서울특별시
- 교육청
- 국토교통부
- 신안산선
- 포스코건설
- 대한민국
- 조선민주주의인민공화국
- 평양직할시
- 정상회담
- 중화인민공화국
- 윈난성
- 대한민국 국회
- 국회의원
- 대한민국
- 경기도
- 파주시
- 통일대교
- 민통선
- JSA
- 대한민국
- 조선민주주의인민공화국
- 8월 13일
- 판문점
- 통일각
- 4·27 판문점 선언
- 조선민주주의인민공화국산 석탄 대한민국 반입 사건
- 대한민국
- 관세청
- 대한민국
- 최저임금
- 8월 21일
- 인천시
- 서울특별시
- 광역버스
- 대한민국
- 워마드

Process finished with exit code 0
~~~
***
# 위 코드를 활용해 모차르트의 곡들을 크롤링하기
~~~Python
from bs4 import BeautifulSoup
import urllib.request as req

url = "https://terms.naver.com/entry.nhn?docId=351953&mobile&cid=51045&categoryId=51045" #url
res = req.urlopen(url)
soup = BeautifulSoup(res, "html.parser")

a_list = soup.select("#size_ct > div.att_type > div > div.wr_tmp_profile > div > table > tbody > tr > td a") #a 태그 전까지의 모든 태그 아래의 모든 a 태그를 모두 선택

for a in a_list:
        name = a.string
        print("-", name)
~~~
> 실행결과

~~~
C:\Users\OHG\PycharmProjects\untitled\venv\Scripts\python.exe C:/Users/OHG/PycharmProjects/untitled/Main.py
- 디베르티멘토 제17번 D장조
- 세레나데 제10번 B플랫장조
- 피아노 협주곡 제9번 E플랫장조
- 피아노 협주곡 제20번 d단조
- 피아노 협주곡 제21번 C장조
- 피아노 협주곡 제23번 A장조
- 피아노 협주곡 제24번 c단조 
- 피아노 협주곡 제27번 B플랫장조 
- 바이올린 협주곡 제3번 G장조
- 플루트 협주곡 제2번
- 클라리넷 협주곡 A장조
- 플루트와 하프를 위한 협주곡 C장조
- 현악 5중주곡 제5번 g단조
- 현악 5중주곡 제6번 E플랫장조
- 현악 4중주곡 제17번 B플랫장조 사냥
- 피아노 4중주곡 제1번 g단조
- 바이올린 소나타 제24번 C장조
- 바이올린 소나타 제28번 e단조
- 바이올린 소나타 B플랫장조
- 피아노 소나타 제8번 a단조
- 피아노 소나타 제13번
- 피아노 소나타 제14번 c단조
- 피아노 소나타 제15번 C장조
- 레퀴엠 d단조

Process finished with exit code 0
~~~
하지만 하이퍼링크 태그인 `<a>`만 조사가 된다. 조금 더 공부가 필요하다.
