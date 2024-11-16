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

![Mass-Spring-Damper System](/images/matlab-10/70117378-ae21f800-16a8-11ea-91a0-b861a296a778.png)
<!-- more -->
![General Solution](/images/matlab-10/70117517-ffca8280-16a8-11ea-874c-742e2bbb72f8.png)

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

![Look up table](/images/matlab-10/70118412-3d300f80-16ab-11ea-9d22-8fc500c3abbc.png)
![display b, k](/images/matlab-10/70118523-82ecd800-16ab-11ea-982c-2b64558e511d.png)
![Wn, Zeta](/images/matlab-10/70118996-a6fce900-16ac-11ea-9c32-43ead6a57246.png)
![Dynamic properties at 32'C](/images/matlab-10/70119141-fb07cd80-16ac-11ea-90ad-7c9e93341126.png)


![Subsystem](/images/matlab-10/70120137-2ab7d500-16af-11ea-8fe3-cc9926cf07b6.png)
![](/images/matlab-10/70124317-de24c780-16b7-11ea-801e-bad594ed62ef.png)

![Pretty](/images/matlab-10/70122875-cb5cc380-16b4-11ea-9795-d315b54971eb.png)

![Scope](/images/matlab-10/70124417-1d531880-16b8-11ea-9acf-d866858e427e.png)