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

~~~xml _config.yml
alias:
    기존 url: 새로운 url
    /2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/: /a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/
~~~

[https://zerohertz.github.io/2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/](https://zerohertz.github.io/2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/)

+ `_config.yml`에 위의 소스를 사용하여 `redirection` 가능
+ `url`은 `local` 위치로 지정