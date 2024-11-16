---
title: MATLAB (5)
date: 2019-11-06 15:00:04
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Control system designer

> SISO System

~~~Matlab
>> controlSystemDesigner(tf(1,[1,2]))
~~~
<!-- more -->
![Control system designer](/images/matlab-5/68272624-95252780-00a7-11ea-9c69-39ae46e2a499.png)
![Architecture](/images/matlab-5/68272880-4deb6680-00a8-11ea-985d-fbfa1b949c75.png)

> Pole, Zero 추가 가능

![](/images/matlab-5/68272777-006ef980-00a8-11ea-972d-07d84a487752.png)

> Simulink export

![](/images/matlab-5/68272831-25fc0300-00a8-11ea-9feb-c365ba1a4911.png)

> Example

~~~Matlab
controlSystemDesigner(tf(1,[1,0,2]))
~~~

![원래의 System](/images/matlab-5/68273645-6e1c2500-00aa-11ea-9ef8-bf99b65d1b54.png)
![Pole과 Zero를 변동시켜 안정화 시키기](/images/matlab-5/68273575-2dbca700-00aa-11ea-817e-4a3356902d52.png)

# Simulink로 PID제어

![](/images/matlab-5/68274014-54c7a880-00ab-11ea-8188-9d9eb3ef1b78.png)
![Tuner](/images/matlab-5/68274332-31e9c400-00ac-11ea-88c7-7254cba4bd2b.png)
