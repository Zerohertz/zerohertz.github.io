---
title: MATLAB (1)
date: 2019-09-18 14:54:14
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Operator

![operator](/images/matlab-1/operator.png)
> `clc` - Interpreter initialization, `help <함수>` - 설명

<!-- more -->

![dataset](/images/matlab-1/dataset.png)
***
# Calculation

![calculation](/images/matlab-1/calculation.png)
***
# Variable

![variable](/images/matlab-1/variable.png)
***
# Trigonometric function

![trigonometric-function](/images/matlab-1/trigonometric-function.png)
***
# Graph plot

~~~Matlab
grid on %Generate grid
bar(b) %Bar graph
xlabel('name')
ylabel('name')
plot(b,'*') %Line style
plot(b, '+')
axis([0 10 0 10]) %Scale
~~~

![plot](/images/matlab-1/plot.png)

![figure](/images/matlab-1/figure.png)

![legend](/images/matlab-1/legend.png)
***
# Text and Charaters

![text-charaters](/images/matlab-1/text-charaters.png)
***
# Array

~~~Matlab
>> magic(4)

ans =

    16     2     3    13
     5    11    10     8
     9     7     6    12
     4    14    15     1

>> help magic
magic - 마방진(Magic Square)

     행과 열의 합계가 동일하고 1 ~ n2 범위의 정수로 생성된 nxn 행렬을 반환합니다.

    M = magic(n)

    참고 항목 ones, rand

    magic에 대한 함수 도움말 페이지

>> ans(4,2) %4행 2

ans =

    14

>> a=magic(4)

a =

    16     2     3    13
     5    11    10     8
     9     7     6    12
     4    14    15     1

>> a(3,:)

ans =

     9     7     6    12

>> a(1:3,2)

ans =

     2
    11
     7
~~~
> `:`는 `from to`의 의미를 지님

***
# Example

+ Matrix

~~~Matlab
a=[1 2 3]
b=[4 5 6]
~~~

![matrix-1](/images/matlab-1/matrix-1.png)

~~~Matlab
a=[[1,2,3];[4,5,6];[7,8,9]]
b=[1 2 3;4 5 6;7 8 9]
~~~

![matrix-2](/images/matlab-1/matrix-2.png)

~~~Matlab
>> A = [1 2 0; 2 5 -1; 4 10 -1]

A =

     1     2     0
     2     5    -1
     4    10    -1

>> B=A'

B =

     1     2     4
     2     5    10
     0    -1    -1

>> C=A*B

C =

     5    12    24
    12    30    59
    24    59   117

>> C=A.*B

C =

     1     4     0
     4    25   -10
     0   -10     1
~~~

+ Graph plot

![graph-1](/images/matlab-1/graph-1.png)
![graph-2](/images/matlab-1/graph-2.png)
