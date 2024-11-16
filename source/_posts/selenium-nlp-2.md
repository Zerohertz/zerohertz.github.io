---
title: Web Crawler for NLP (2)
date: 2018-09-20 13:01:41
categories:
- Etc.
tags:
- Python
- Selenium
---
# Pytube
~~~Python
Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:59:51) [MSC v.1914 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pytube
>>> yt = pytube.YouTube("https://www.youtube.com/watch?v=94YwFIJ-yR0&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3&index=1")
>>> vids = yt.streams.all()
>>> for i in range(len(vids)):
...     print(i,':',vids[i])
...
0 : <Stream: itag="22" mime_type="video/mp4" res="720p" fps="30fps" vcodec="avc1.64001F" acodec="mp4a.40.2">
1 : <Stream: itag="43" mime_type="video/webm" res="360p" fps="30fps" vcodec="vp8.0" acodec="vorbis">
2 : <Stream: itag="18" mime_type="video/mp4" res="360p" fps="30fps" vcodec="avc1.42001E" acodec="mp4a.40.2">
3 : <Stream: itag="36" mime_type="video/3gpp" res="240p" fps="30fps" vcodec="mp4v.20.3" acodec="mp4a.40.2">
4 : <Stream: itag="17" mime_type="video/3gpp" res="144p" fps="30fps" vcodec="mp4v.20.3" acodec="mp4a.40.2">
5 : <Stream: itag="136" mime_type="video/mp4" res="720p" fps="30fps" vcodec="avc1.4d401f">
6 : <Stream: itag="247" mime_type="video/webm" res="720p" fps="30fps" vcodec="vp9">
7 : <Stream: itag="135" mime_type="video/mp4" res="480p" fps="30fps" vcodec="avc1.4d401f">
8 : <Stream: itag="244" mime_type="video/webm" res="480p" fps="30fps" vcodec="vp9">
9 : <Stream: itag="134" mime_type="video/mp4" res="360p" fps="30fps" vcodec="avc1.4d401e">
10 : <Stream: itag="243" mime_type="video/webm" res="360p" fps="30fps" vcodec="vp9">
11 : <Stream: itag="133" mime_type="video/mp4" res="240p" fps="30fps" vcodec="avc1.4d4015">
12 : <Stream: itag="242" mime_type="video/webm" res="240p" fps="30fps" vcodec="vp9">
13 : <Stream: itag="160" mime_type="video/mp4" res="144p" fps="30fps" vcodec="avc1.4d400c">
14 : <Stream: itag="278" mime_type="video/webm" res="144p" fps="30fps" vcodec="vp9">
15 : <Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">
16 : <Stream: itag="171" mime_type="audio/webm" abr="128kbps" acodec="vorbis">
17 : <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">
18 : <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">
19 : <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">
~~~
영상 정보 퍼오기
<!-- more -->
***
# 음성 정보만 퍼오기
~~~Python
from selenium import webdriver
import os
import time
import pytube

browser = webdriver.Chrome('C:/Users/OHG/Downloads/chromedriver_win32/chromedriver')  #Directory지정
browser.implicitly_wait(5)  #암묵적으로 3초 딜레이
url = "https://www.youtube.com/watch?v=94YwFIJ-yR0&list=PL3Eb1N33oAXijqFKrO83hDEN0HPwaecV3&index=1"
browser.get(url)

for a in range(1, 100): #url의 개수
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
        stream = yt.streams.all()
        stream[15].download(os.getcwd(), "내용")
        browser.save_screenshot("Website.png")
        browser.find_element_by_css_selector('#movie_player > div.ytp-chrome-bottom > div.ytp-chrome-controls > div.ytp-left-controls > a.ytp-next-button.ytp-button').click()
        os.chdir("..")
        time.sleep(3)
browser.quit()
~~~
***
# 대사 issue
> 알고보니 YouTube의 글들은 실제 대사가 아니였다

따라서 YouTube에 있는 글중 본문링크로 들어가서 대사를 퍼와야한다