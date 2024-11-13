---
title: MATLAB (2)
date: 2019-10-02 14:35:16
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Bar Plots

~~~Matlab
x = -2.9:0.2:2.9;
y = exp(-x.*x);
bar(x,y)
~~~
***
# Stairstep Plots

~~~Matlab
x = 0:0.25:10;
y = sin(x);
stairs(x, y)
~~~
<!-- more -->
***
# Errorbar Plots

~~~Matlab
x = -2:0.1:2;
y = erf(x);
eb = rand(size(x))/7;
errorbar(x,y,eb)
~~~
***
# Polar Plots

~~~Matlab
theta = 0:0.01:2*pi; %angle
rho = abs(sin(2*theta).*cos(2*theta)); %radius
polarplot(theta,rho)
~~~
***
# Stem plots

~~~Matlab
x = 0:0.1:4;
y = sin(x.^2).*exp(-x);
stem(x,y)
~~~
***
# Scatter plots

~~~Matlab
load patients Height Weight Systolic %load data
scatter(Height,Weight) %scatter plot of Weight vs. Height
xlabel('Height')
ylabel('Weight')
~~~
***
# 부분분수 전개

~~~Matlab
b=[-4 8]; %분자
a=[1 6 8]; %분모
[r,p,k]=residue(b,a)
~~~

~~~Matlab
r=[-12 8];
p=[-4 -2];
k=[];
[b,a] = residue(r,p,k) %역변환
~~~
***
# ODE

~~~Matlab
x=sym('x');
U1=diff((2*x^2-x)^3);
U2=diff((x/(x+1))^2);
~~~
***
# Solutions of ODE

~~~Matlab
syms x y;
U=dsolve('D2y+2*Dy+3*y=0','y(0)=1','Dy(0)=1','x')
~~~