---
title: MATLAB (8)
date: 2019-11-27 14:09:33
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Lead-Lag Compensation

~~~Matlab
>> num=[4];
>> den=[1 0.5 0];
>> v=[-5 5 -5 5];
>> axis(v);
~~~
<!-- more -->
![axis](/images/matlab-8/69695446-be255f00-111f-11ea-9e78-f495fbffbdd6.png)

~~~Matlab
>> rlocus(num,den);
>> axis(v);
~~~

![rlocus](/images/matlab-8/69695546-13fa0700-1120-11ea-95ac-99820a5440ed.png)
![axis](/images/matlab-8/69695563-26744080-1120-11ea-9d34-37df355910ea.png)

~~~Matlab
>> rlocus(num,den);
>> grid on
>> hold on
~~~

![grid on](/images/matlab-8/69695770-d2b62700-1120-11ea-819a-a6d7136fe748.png)

~~~Matlab
>> numc=[25.04 5.008];
>> denc=[1 5.03247 0.0626 0];
>> rlocus(numc,denc);
~~~

![Lead-Lag Compensation](/images/matlab-8/69695807-f7120380-1120-11ea-888c-f17dcf041f75.png)

~~~Matlab
>> den=[1 0.5 4];
>> denc=[1 5.0327 25.1026 5.008];
>> t=0:0.1:10;
>> step(num,den,t);
>> hold on
>> step(numc,denc,t);
>> grid on
~~~

![Step Response](/images/matlab-8/69696051-d302f200-1121-11ea-8636-644a8b8a04c6.png)

~~~Matlab
>> step(num,den,t);
>> den=[1 0.5 4 0];
>> step(num,den,t);
>> hold on
>> denc=[1,5.0327 25.1026 5.008 0];
>> step(numc,denc,t);
>> plot(t,t,'k');
>> grid on
~~~

![Ramp Response](/images/matlab-8/69696516-40635280-1123-11ea-981e-32df9423b410.png)