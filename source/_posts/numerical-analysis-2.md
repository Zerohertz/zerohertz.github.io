---
title: Numerical Analysis (2)
date: 2018-10-01 20:07:02
categories:
- Etc.
tags:
- Python
- B.S. Course Work
---
# 한번에 그리기
~~~Python
import plotly
import math
from plotly.graph_objs import Scatter, Layout



t1 = 0 #변수, 상수 선언
t2 = 0
t3 = 0
t4 = 0
g = 9.81
c = 12.5
m = 68.1
delta1 = 0.6
delta2 = 1.2
delta3 = 2.4
vel = 0
vel1 = 0
vel2 = 0
vel3 = 0
vel4 = 0
Ti1 = [0]
Ve1 = [0]
Ti2 = [0]
Ve2 = [0]
Ti3 = [0]
Ve3 = [0]
Ti4 = [0]
Ve4 = [0]


def velocity(vel, delta):
    value = vel + (g - c * vel / m) * delta #식의 해
    return value

def realsol(t):
    Value = 53.44*(1-math.exp(1)**(-0.18355*t))
    return Value


for a in range(0, 80): #몇번 반복할지
    vel1 = velocity(vel1, delta1)
    t1 = t1 + delta1
    T1 = round(t1, 1) #유효숫자
    VEL1 = round(vel1, 1)
    Ti1 = Ti1 + [T1]
    Ve1 = Ve1 + [VEL1]

for a in range(0, 40): #몇번 반복할지
    vel2 = velocity(vel2, delta2)
    t2 = t2 + delta2
    T2 = round(t2, 1) #유효숫자
    VEL2 = round(vel2, 1)
    Ti2 = Ti2 + [T2]
    Ve2 = Ve2 + [VEL2]

for a in range(0, 20): #몇번 반복할지
    vel3 = velocity(vel3, delta3)
    t3 = t3 + delta3
    T3 = round(t3, 1) #유효숫자
    VEL3 = round(vel3, 1)
    Ti3 = Ti3 + [T3]
    Ve3 = Ve3 + [VEL3]

for a in range(0, 4800): #몇번 반복할지
    vel4 = realsol(t4)
    t4 = t4 + 0.01
    T4 = round(t4, 1) #유효숫자
    VEL4 = round(vel4, 1)
    Ti4 = Ti4 + [T4]
    Ve4 = Ve4 + [VEL4]


plotly.offline.plot({
    "data": [Scatter(x=Ti1, y=Ve1)],
    "layout": Layout(title="ODE 풀이, delta = 0.6")
})

plotly.offline.plot({
    "data": [Scatter(x=Ti2, y=Ve2)],
    "layout": Layout(title="ODE 풀이, delta = 1.2")
})

plotly.offline.plot({
    "data": [Scatter(x=Ti3, y=Ve3)],
    "layout": Layout(title="ODE 풀이, delta = 2.4")
})

plotly.offline.plot({
    "data": [Scatter(x=Ti4, y=Ve4)],
    "layout": Layout(title="ODE 풀이, realsol")
})
~~~
<!-- more -->
![실행결과](/images/numerical-analysis-2/46286543-7ad26d80-c5ba-11e8-9c8b-48a847db3c5f.png)
***
# Error 추가

~~~Python
import plotly
import math
from plotly.graph_objs import Scatter, Layout



t1 = 0 #변수, 상수 선언
t2 = 0
t3 = 0
t4 = 0
g = 9.81
c = 12.5
m = 68.1
delta1 = 0.6
delta2 = 1.2
delta3 = 2.4
vel = 0
vel1 = 0
vel2 = 0
vel3 = 0
vel4 = 0
Ti1 = [0]
Ve1 = [0]
Ti2 = [0]
Ve2 = [0]
Ti3 = [0]
Ve3 = [0]
Ti4 = [0]
Ve4 = [0]


def velocity(vel, delta):
    value = vel + (g - c * vel / m) * delta #식의 해
    return value

def realsol(t):
    Value = 53.44*(1-math.exp(1)**(-0.18355*t))
    return Value

def Error1():
    Err1 = (realsol(12) - Ve1[20])*100 / realsol(12)
    round(Err1, 1)
    return Err1

def Error2():
    Err2 = (realsol(12) - Ve2[10])*100 / realsol(12)
    round(Err2, 1)
    return Err2

def Error3():
    Err3 = (realsol(12) - Ve3[5])*100 / realsol(12)
    round(Err3, 1)
    return Err3


for a in range(0, 80): #몇번 반복할지
    vel1 = velocity(vel1, delta1)
    t1 = t1 + delta1
    T1 = round(t1, 1) #유효숫자
    VEL1 = round(vel1, 1)
    Ti1 = Ti1 + [T1]
    Ve1 = Ve1 + [VEL1]

for a in range(0, 40): #몇번 반복할지
    vel2 = velocity(vel2, delta2)
    t2 = t2 + delta2
    T2 = round(t2, 1) #유효숫자
    VEL2 = round(vel2, 1)
    Ti2 = Ti2 + [T2]
    Ve2 = Ve2 + [VEL2]

for a in range(0, 20): #몇번 반복할지
    vel3 = velocity(vel3, delta3)
    t3 = t3 + delta3
    T3 = round(t3, 1) #유효숫자
    VEL3 = round(vel3, 1)
    Ti3 = Ti3 + [T3]
    Ve3 = Ve3 + [VEL3]

for a in range(0, 4800): #몇번 반복할지
    vel4 = realsol(t4)
    t4 = t4 + 0.01
    T4 = round(t4, 1) #유효숫자
    VEL4 = round(vel4, 1)
    Ti4 = Ti4 + [T4]
    Ve4 = Ve4 + [VEL4]


plotly.offline.plot({
    "data": [Scatter(x=Ti1, y=Ve1)],
    "layout": Layout(title="ODE 풀이, delta = 0.6")
})

plotly.offline.plot({
    "data": [Scatter(x=Ti2, y=Ve2)],
    "layout": Layout(title="ODE 풀이, delta = 1.2")
})

plotly.offline.plot({
    "data": [Scatter(x=Ti3, y=Ve3)],
    "layout": Layout(title="ODE 풀이, delta = 2.4")
})

plotly.offline.plot({
    "data": [Scatter(x=Ti4, y=Ve4)],
    "layout": Layout(title="ODE 풀이, realsol")
})

print(Error1())
print(Error2())
print(Error3())
~~~
> 실행결과

~~~Python
-1.6113612675083666
-3.0839896916751606
-6.239622029175421
~~~