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
![control-system-designer](/images/matlab-5/control-system-designer.png)
![architecture](/images/matlab-5/architecture.png)

> Pole, Zero 추가 가능

![controlsystemdesigner](/images/matlab-5/controlsystemdesigner.png)

> Simulink export

![simulink-1](/images/matlab-5/simulink-1.png)

> Example

~~~Matlab
controlSystemDesigner(tf(1,[1,0,2]))
~~~

![system](/images/matlab-5/system.png)
![pole-zero](/images/matlab-5/pole-zero.png)

# Simulink로 PID제어

![simulink-2](/images/matlab-5/simulink-2.png)
![tuner](/images/matlab-5/tuner.png)
