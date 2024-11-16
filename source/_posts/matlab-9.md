---
title: MATLAB (9)
date: 2019-11-27 14:59:48
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Frequency Response

## Natural frequency
+ `wn` = sqrt(k/m)(rad/s) - `회전 주파수`
+ `fn` = wn/(2*pi)(Hz) - `주파수`

~~~Matlab
>> m=10;
>> b=0.1;
>> k=1000;
>> wn=sqrt(k/m)

wn =

    10
~~~
<!-- more -->

## Definition of dB

`dB = -20log(Output/Input)`

## Bode plot

~~~Matlab
num=[1];
den=[10 0.1 1000];
sys=tf(num,den);
bode(sys)
~~~

![bode-plot-1](/images/matlab-9/bode-plot-1.png)

+ Hz

~~~Matlab
num=[1];
den=[10 0.1 1000];
sys=tf(num,den);
bode(sys)
figure
h=bodeplot(sys);
setoptions(h,'FreqUnits','Hz');
~~~

> k

~~~Matlab
num=[1];
den=[10 0.1 1000];
den=[10 0.1 4000];
~~~

![bode-plot-2](/images/matlab-9/bode-plot-2.png)

+ b is same in frequency domain