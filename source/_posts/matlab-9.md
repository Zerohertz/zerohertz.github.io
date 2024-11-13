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

![](https://user-images.githubusercontent.com/42334717/69698862-caaeb500-1129-11ea-91bb-9a5c5defa953.png)

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

>k

~~~Matlab
num=[1];
den=[10 0.1 1000];
den=[10 0.1 4000];
~~~

![](https://user-images.githubusercontent.com/42334717/69700035-d059ca00-112c-11ea-905e-5f83d38f2006.png)

+ b is same in frequency domain