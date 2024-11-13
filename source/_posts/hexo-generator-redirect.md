---
title: hexo-generator-redirect
date: 2020-08-20 20:46:01
categories:
- Etc.
tags:
- Hexo
---
# Introduction

`Hexo`에서 `url`을 `redirection`해주는 플러그인

# Install

```shell
$ sudo npm install hexo-generator-redirect --save
```

<!-- More -->

# Use

~~~ejs /layout/redirect.ejs
<% const newUrl = full_url_for(page.target.path) %>

<h1>Page address was changed</h1>
<p>The new page address is <a href="<%= newUrl %>"><%= newUrl %></a></p>

<script type="text/javascript">
  setTimeout(function(){ document.location.href = '<%= newUrl %>'; }, 1000);
</script>
~~~

~~~md example.md
title: >-
  A PHM Approach to Additive Manufacturing Equipment Health Monitoring, Fault
  Diagnosis, and Quality Control
date: 2020-01-30 09:43:24
redirect_from:
- /2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/
~~~

[https://zerohertz.github.io/2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/](https://zerohertz.github.io/2020/01/30/a-phm-approach-to-additive-manufacturing-equipment-health-monitoring-fault-diagnosis-and-quality-control/)