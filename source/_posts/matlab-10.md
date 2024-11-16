---
title: MATLAB (10)
date: 2019-12-04 15:10:01
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Mass-Spring-Damper System

![mass-spring-damper-system](/images/matlab-10/mass-spring-damper-system.png)
<!-- more -->
![general-solution](/images/matlab-10/general-solution.png)

>General Solution

~~~Matlab
syms s x0 xp0 wn z;
Xs=(x0*(s+z*wn)+(xp0+z*wn*x0)/(wn*(sqrt(1-z^2)))*wn*sqrt(1- z^2))/((s+z*wn)^2+(wn*sqrt(1-z^2))^2)
U=ilaplace(Xs)
~~~

>실행결과

~~~
Xs =
 
-(xp0 + x0*(s + wn*z) + wn*x0*z)/(wn^2*(z^2 - 1) - (s + wn*z)^2)
 
 
U =
 
x0*exp(-t*wn*z)*(cosh(t*wn*(z^2 - 1)^(1/2)) - (sinh(t*wn*(z^2 - 1)^(1/2))*(wn*z - (xp0 + 2*wn*x0*z)/x0))/(wn*(z^2 - 1)^(1/2)))
~~~
***
# Simulink

![look-up-table](/images/matlab-10/look-up-table.png)
![display-b-k](/images/matlab-10/display-b-k.png)
![wn-zeta](/images/matlab-10/wn-zeta.png)
![dynamic-properties-at-32'c](/images/matlab-10/dynamic-properties-at-32'c.png)


![subsystem](/images/matlab-10/subsystem.png)
![model](/images/matlab-10/model.png)

![pretty](/images/matlab-10/pretty.png)

![scope](/images/matlab-10/scope.png)