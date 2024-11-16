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

![](/images/matlab-1/65117288-c2991180-da24-11e9-9e58-62a12172c887.png)
> `clc` - Interpreter initialization, `help <함수>` - 설명

<!-- more -->

![Dataset 저장](/images/matlab-1/65118683-54ede500-da26-11e9-9e8e-f1d9e4b20afa.png)
***
# Calculation

![](/images/matlab-1/65117331-cfb60080-da24-11e9-96db-d0adbdd233e2.png)
***
# Variable

![](/images/matlab-1/65120630-e52d2980-da28-11e9-9775-c5ad78882032.png)
***
# Trigonometric function

![](/images/matlab-1/65120603-db0b2b00-da28-11e9-96b8-3ecbaf0f9334.png)
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

![plot(b,'*'), axis([0 10 0 10])](/images/matlab-1/65121979-08f16f00-da2b-11e9-840d-f449a083cc08.png)

![figure](/images/matlab-1/65123019-5373eb00-da2d-11e9-9d4f-677c5dde37e7.png)

![legend('name')](/images/matlab-1/65123304-e44ac680-da2d-11e9-98ac-1cc502d5836d.png)
***
# Text and Charaters

![](/images/matlab-1/65122267-78675e80-da2b-11e9-9717-a449aebed828.png)
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

![](/images/matlab-1/65119952-de51e700-da27-11e9-8e3c-21985e85a82e.png)

~~~Matlab
a=[[1,2,3];[4,5,6];[7,8,9]]
b=[1 2 3;4 5 6;7 8 9]
~~~

![](/images/matlab-1/65120263-4ef90380-da28-11e9-86aa-2008cf74cda9.png)

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

![Figure 복사](/images/matlab-1/65121022-8320f400-da29-11e9-9017-0ff700f8a56a.png)
![](/images/matlab-1/65120903-5a006380-da29-11e9-9b4b-0c0320ffd245.png)
