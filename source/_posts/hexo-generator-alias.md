---
title: hexo-generator-alias
date: 2020-08-20 12:04:41
categories:
- Etc.
tags:
- Hexo
---
# Introduction

`Hexo`에서 `url`을 `redirection`해주는 플러그인

# Install

```shell
$ sudo npm install hexo-generator-alias --save
```

<!-- More -->

# Use

> 검색 시 기존의 url을 노출

![기존](/images/hexo-generator-alias/기존.png)

> 404

![404](/images/hexo-generator-alias/404.png)

~~~yaml _config.yml
alias:
  /apache-airflow-1/: /airflow-1/
  /apache-airflow-2/: /airflow-2/
  /analysis-of-many-bodies-in-one-part-with-ansys-act/: /ansys-act/
...
~~~

[https://zerohertz.github.io/apache-airflow-1/](https://zerohertz.github.io/apache-airflow-1/)

+ `_config.yml`에 위의 소스를 사용하여 `redirection` 가능
+ `url`은 `local` 위치로 지정