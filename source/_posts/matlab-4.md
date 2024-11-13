---
title: MATLAB (4)
date: 2019-10-16 15:07:36
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Transfer Function

> tf?

~~~Matlab
>> help tf
tf - Transfer function model

    Use tf to create real-valued or complex-valued transfer function models, or to
    convert dynamic system models to transfer function form.

    sys = tf(numerator,denominator)
    sys = tf(numerator,denominator,ts)
    sys = tf(numerator,denominator,ltiSys)
    sys = tf(m)
    sys = tf(___,Name,Value)
    sys = tf(ltiSys)
    sys = tf(ltiSys,component)
    s = tf('s')
    z = tf('z',ts)
~~~
<!-- more -->
> Make Transfer function

~~~Matlab
>> a = [1 0]

a =

     1     0

>> b = [1 2 1]

b =

     1     2     1

>> sys = tf(a,b)

sys =
 
        s
  -------------
  s^2 + 2 s + 1
 
Continuous-time transfer function.
~~~

> pole?

~~~Matlab
>> help pole
--- pole에 대한 도움말 ---

pole - Poles of dynamic system

    This MATLAB function returns the poles of the SISO or MIMO dynamic system model
    sys.

    P = pole(sys)
    P = pole(sys,J1,...,JN)
~~~

> Find Poles

~~~Matlab
>> P = pole(sys)

P =

    -1
    -1
~~~
***
# Graph Plot

> Legend

~~~Matlab
>> help legend
legend - 좌표축에 범례 추가

     플로팅된 각 데이터 계열에 대한 설명 레이블을 포함한 범례를 만듭니다.

    legend
    legend(label1,...,labelN)
    legend(labels)
    legend(subset,___)
    legend(target,___)
    legend(___,'Location',lcn)
    legend(___,'Orientation',ornt)
    legend(___,Name,Value)
    legend(bkgd)
    lgd = legend(___)
    legend(vsbl)
    legend('off')
~~~