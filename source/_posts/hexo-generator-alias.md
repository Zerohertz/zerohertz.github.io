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

![기존](https://user-images.githubusercontent.com/42334717/90713568-95ae4800-e2e0-11ea-8856-5a62f01e517e.png)

> 404

![404](https://user-images.githubusercontent.com/42334717/90713667-cf7f4e80-e2e0-11ea-8e78-2bc71f7c62cd.png)

~~~xml _config.yml
alias:
    기존 url: 새로운 url
    /2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/: /a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/
~~~

[https://zerohertz.github.io/2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/](https://zerohertz.github.io/2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/)

+ `_config.yml`에 위의 소스를 사용하여 `redirection` 가능
+ `url`은 `local` 위치로 지정