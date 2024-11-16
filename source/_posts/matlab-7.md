---
title: MATLAB (7)
date: 2019-11-24 22:19:10
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work

---
# TF&state-space
 
~~~Matlab
syms Y s y(t) t
 
laplace(10*diff(diff(y(t)))+0.4*diff(y(t))+4*y(t))
Y1=s*Y;
Y2=s*Y1;
sol=solve(10*Y2+0.4*Y1+4*Y-1,Y);
pretty(sol)
 
num=[1];
den=[10 0.4 4];
h=tf(num, den);
[A,B,C,D]=tf2ss(num, den)
~~~
<!-- more -->

>실행결과
~~~Matlab
>> Untitled
 
ans =
 
(2*s*laplace(y(t), t, s))/5 - (2*y(0))/5 - 10*s*y(0) + 10*s^2*laplace(y(t), t, s) - 10*subs(diff(y(t), t), t, 0) + 4*laplace(y(t), t, s)
 
       1
---------------
    2   2 s
10 s  + --- + 4
         5
 
 
A =
 
   -0.0400   -0.4000
    1.0000         0
 
 
B =
 
     1
     0
 
 
C =
 
         0    0.1000
 
 
D =
 
     0
~~~
*** 
# rand, polyfit
 
~~~Matlab
xran=rand(11,1);
plot(xran)
xlabel('Time[s]')
ylabel('Mag.[mm]')
x=0:10;
p=polyfit(x',xran,3);
v=diff(p)
a=diff(v)
~~~
 
>실행결과
 
~~~Matlab 
v =
 
   -0.1215    0.6093   -0.5209
 
 
a =
 
    0.7309   -1.1302
~~~
***
# Constant
 
~~~Matlab
x=[1:1:4];
y=[2 3 2 1];
plot(y)
xlabel('Time[s]')
ylabel('Mag.[mm]')
p=polyfit(x,y,3);
v=diff(p)
a=diff(v)
~~~
 
>실행결과
 
~~~Matlab
v =
 
   -3.3333   10.6667  -10.6667
 
 
a =
 
   14.0000  -21.3333
~~~
***
# Simulink

![State-space equation, Scope](/images/matlab-7/69495291-b50e7500-0f08-11ea-867f-47081a69eaef.png)