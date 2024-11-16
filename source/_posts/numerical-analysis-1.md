---
title: Numerical Analysis (1)
date: 2018-09-10 18:36:00
categories:
- Etc.
tags:
- Python
- B.S. Course Work
---
# 중력에 대한 ODE를 Numerical analysis
~~~Python
t = 0 #변수, 상수 선언
g = 9.81
c = 12.5
m = 68.1
delta = 2.4
vel = 0


def velocity(t):
    value = vel + (g - c * vel / m) * delta #식의 해
    return value


print(t, vel)

for a in range(1, 100): #몇번 반복할지
    vel = velocity(t)
    t = t + delta
    print(t, vel)
~~~

~~~
0 0
2.4 23.544
4.8 36.71619383259912
7.199999999999999 44.085659104581886
9.6 48.208663904325554
12.0 50.51536703017333
14.4 51.80590137811459
#중간생략 왼쪽이 시간, 오른쪽이 Velocity
232.80000000000038 53.44487999999999
235.2000000000004 53.44487999999999
237.6000000000004 53.44487999999999
~~~
<!-- more -->
시간이 `언더플로우` 되는것 같다
`Graph`를 그리는 법을 찾아보자!
***
# Mantissa Issue
위에 `언더플로우`라고 생각했던건 실수를 표시할때 컴퓨터에 생기는 문제였다. 정수와 상대적으로 표시할 부분이 작아지므로 차이가 난다. 또한 `유효숫자`를 표시하는 법을 만들어야한다.
~~~Python
t = 0 #변수, 상수 선언
g = 9.81
c = 12.5
m = 68.1
delta = 0.24e1
vel = 0


def velocity(t):
    value = vel + (g - c * vel / m) * delta #식의 해
    return value


print(t, vel)

for a in range(1, 100): #몇번 반복할지
    vel = velocity(t)
    t = t + delta
    T = round(t, 1) #유효숫자
    VEL = round(vel, 1)
    print(T, VEL)
~~~

~~~
0 0
2.4 23.5
4.8 36.7
7.2 44.1
9.6 48.2
12.0 50.5
14.4 51.8
16.8 52.5
19.2 52.9
21.6 53.2
#중간생략 왼쪽이 시간, 오른쪽이 Velocity
220.8 53.4
223.2 53.4
225.6 53.4
228.0 53.4
230.4 53.4
232.8 53.4
235.2 53.4
237.6 53.4
~~~
***
# Graph plot
참고 : [그래프 그리기](https://zzsza.github.io/development/2018/08/24/data-visualization-in-python/)
~~~Python
import plotly

from plotly.graph_objs import Scatter, Layout



t = 0 #변수, 상수 선언
g = 9.81
c = 12.5
m = 68.1
delta = 0.24e1
vel = 0
Ti = [0]
Ve = [0]

def velocity(t):
    value = vel + (g - c * vel / m) * delta #식의 해
    return value


for a in range(0, 20): #몇번 반복할지
    vel = velocity(t)
    t = t + delta
    T = round(t, 1) #유효숫자
    VEL = round(vel, 1)
    Ti = Ti + [T]
    Ve = Ve + [VEL]

print(Ti)
print(Ve)


plotly.offline.plot({
    "data": [Scatter(x=Ti, y=Ve)],
    "layout": Layout(title="ODE 풀이")
})
~~~
![실행결과](/images/numerical-analysis-1/45418251-f37e9200-b6be-11e8-9b1c-9648e01a6cf2.png)
***
# input까지
~~~Python
import plotly

from plotly.graph_objs import Scatter, Layout



t = 0 #변수, 상수 선언
g = 9.81
c = 12.5
m = 68.1
delta = input("시간의 변화량 delta를 입력하세요 : ")
delta = float(delta)
vel = 0
Ti = [0]
Ve = [0]

if delta == 0.6:
    R = 80
elif delta == 1.2:
    R = 40
elif delta == 2.4:
    R = 20

def velocity(t):
    value = vel + (g - c * vel / m) * delta #식의 해
    return value


for a in range(0, R): #몇번 반복할지
    vel = velocity(t)
    t = t + delta
    T = round(t, 1) #유효숫자
    VEL = round(vel, 1)
    Ti = Ti + [T]
    Ve = Ve + [VEL]

print(Ti)
print(Ve)


plotly.offline.plot({
    "data": [Scatter(x=Ti, y=Ve)],
    "layout": Layout(title="ODE 풀이")
})
~~~