---
title: hexo-tag-gdemo
date: 2020-07-28 22:01:03
categories:
- Etc.
tags:
- Hexo
---
# 멋진 Plug in

```shell
$ npm install @heowc/hexo-tag-gdemo
```

<!-- More -->

***

# 사용법

~~~shell
{% gdemo_terminal 'command1;command2;...' 'minHeight' 'windowTitle' 'onCompleteDelay' 'promptString' 'id' 'highlightingLang' %}
content
{% endgdemo_terminal %}
~~~

***

# Example

```shell
{% gdemo_terminal '2017.03 ~ : Konkuk Univ. Mechanical Engineering;2018.06 ~ 2019.11 : Former undergraduate researcher at MRV Lab.(Medical Robotics and Virtual Reality Laboratory);2019.11 ~ : Undergraduate researcher at SiM Lab.(Smart intelligent Manufacturing system Laboratory)' '150px' 'Career' '300' '$' 'career' 'vim' %}
{% endgdemo_terminal %}
```

~~~shell
{% gdemo_terminal '2017.03 ~ : Konkuk Univ. Mechanical Engineering;2018.06 ~ 2019.11 : Former undergraduate researcher at MRV Lab.(Medical Robotics and Virtual Reality Laboratory);2019.11 ~ : Undergraduate researcher at SiM Lab.(Smart intelligent Manufacturing system Laboratory)' '150px' 'Career' '300' '$' 'career' 'vim' %}
{% endgdemo_terminal %}
~~~

[Reference](https://heowc.dev/2018/11/14/introduction-hexo-tag-gdemo/)