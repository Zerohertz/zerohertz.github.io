---
title: Upload Hexo by git
date: 2018-08-19 20:03:41
categories:
- Etc.
tags:
- GitHub
---
## 첫 업로드
~~~
$ git init
$ git remote add origin https://github.com/Zerohertz/Hexo_Blog.git    (github의 주소)
$ git add .
$ git remote -v   (확인)
$ git commit -m "내용"  (주석)
$ git push origin master
~~~
***
## 수정, 추가 업로드
```
$ git add .
$ git commit -m "내용"
$ git push origin master
```