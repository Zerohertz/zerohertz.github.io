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
![axis-1](/images/matlab-8/axis-1.png)

~~~Matlab
>> rlocus(num,den);
>> axis(v);
~~~

![rlocus](/images/matlab-8/rlocus.png)
![axis-2](/images/matlab-8/axis-2.png)

~~~Matlab
>> rlocus(num,den);
>> grid on
>> hold on
~~~

![grid-on](/images/matlab-8/grid-on.png)

~~~Matlab
>> numc=[25.04 5.008];
>> denc=[1 5.03247 0.0626 0];
>> rlocus(numc,denc);
~~~

![lead-lag-compensation](/images/matlab-8/lead-lag-compensation.png)

~~~Matlab
>> den=[1 0.5 4];
>> denc=[1 5.0327 25.1026 5.008];
>> t=0:0.1:10;
>> step(num,den,t);
>> hold on
>> step(numc,denc,t);
>> grid on
~~~

![step-response](/images/matlab-8/step-response.png)

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

![ramp-response](/images/matlab-8/ramp-response.png)