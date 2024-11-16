---
title: Web Crawling by BeautifulSoup (1)
date: 2018-08-16 03:42:19
categories:
- Etc.
tags:
- Python
- BeautifulSoup
---
# BeautifulSoup란?
HTML과 XML을 분석해주는 라이브러리.
***
# find_all() 메서드로 `<a>` 태그 추출하기
```python
from bs4 import BeautifulSoup

html = """
<html><body>
    <ul>
        <li><a href="http://www.naver.com">naver</a></li>
    </ul>
</body></html>
"""

soup = BeautifulSoup(html, 'html.parser')

links = soup.find_all("a")

for a in links:
    href = a.attrs['href']
    text = a.string
    print(text, ">", href)
```
