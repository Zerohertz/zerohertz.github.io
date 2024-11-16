---
title: TensorFlow (1)
date: 2019-08-12 11:21:44
categories:
- 5. Machine Learning
tags:
- Python
- TensorFlow
---
# TensorFlow를 이용한 간단한 곱셈 프로그램

~~~Python
import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)

sess = tf.Session()

print(sess.run(y, feed_dict={a: 3, b: 3}))
~~~
<!-- more -->

+ 텐서플로우 파이썬 모듈 임포트
+ 프로그램 실행 중 값을 변경할 수 있는 `placeholder` 심볼릭 변수 정의
+ `tf.multiply`는 텐서를 조작하기 위해 텐서플로우가 제공하는 곱셈 함수
+ 텐서는 동적 사이즈를 갖는 다차원 데이터 배열
+ `Session()`함수를 통해 세션을 생성함으로써 프로그램이 텐서플로우 라이브러리와 상호작용
+ 세션을 정하여 `run()`메서드를 호출할 때 심볼릭 코드가 실제 실행
+ `run()`메서드에 feed_dict 인자로 변수의 값을 넘김
+ 입력된 수식이 계삭되면 곱의 결과 9를 프린트

![results-1](/images/tensorflow-1/results-1.png)

> 텐서플로우 프로그램의 일반적인 구조 : 전체 알고리즘을 먼저 기술하고 세션을 생성하여 연산을 실행
***
# 간단한 선형 회귀분석(Linear Regression)

+ 변수들 사이의 관계를 분석
+ 알고리즘의 개념이 복잡하지 않고 다양한 문제에 폭 넓게 적용

~~~Python
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []

for i in range(num_points):
         x1 = np.random.normal(0.0, 0.55)
         y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()
~~~

![results-2](/images/tensorflow-1/results-2.png)

+ 코드에서 볼 수 있듯 `y=0.1*x+0.3` 관계를 가지는 데이터를 생성
+ 정규분포를 따라 약간의 편차를 두어 완전히 직선에 일치하지 않는 예시
+ 위 데이터를 모델을 만들기 위한 학습 데이터로 사용

## 코스트 함수

+ `x_data`를 이용해 출력 값 `y_data`를 예측할 수 있는 최적의 파라미터 `W`와 `b`를 찾도록 수정
+ 반복이 일어날 때 마다 개선되고 있는지 확인하기 위해 얼마나 좋은 직선인지를 측정하는 코스트 함수(혹은 에러 함수)를 정의
+ 코스트 함수로 평균제곱오차(mean square error) 사용

~~~Python
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
~~~

+ `Variable` 메소드를 호출하면 텐서플로우 내부의 그래프 데이터 구조에 만들어질 하나의 변수 정의

~~~Python
loss = tf.reduce_mean(tf.square(y - y_data))
~~~

+ 실제 값과 `y=W*x+b`로 계산된 값 간의 거리를 기반으로 코스트 함수 정의
+ 제곱을 하고 모두 더하여 평균
+ 이미 알고 있는 값 `y_data`와 입력 데이터 `x_data`로 계산된 y값 사이의 거리를 제곱한 것의 평균

## 그래디언트 디센트

~~~Python
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
~~~

+ 에러 함수를 최소화하면 데이터에 가장 최적화 된 모델
+ 일련의 파라미터로 된 함수가 주어지면 초기 시작점에서 함수의 값이 최소화 되는 방향으로 파라미터를 변경하는 것을 반복적으로 수행
+ 함수의 기울기가 음의 방향인 쪽으로 진행하면서 반복적으로 최적화를 수행
+ 보통 양의 값을 만들기 위해 거리 값을 제곱
+ 기울기를 계산해야 하므로 에러 함수는 미분 가능

> 알고리즘 실행

+ 아직은 텐서플로우 라이브러리를 호출하는 코드는 단지 내부 그래프 구조에 정보를 추가시킨 것일 뿐 텐서플로우의 실행 모듈은 아직 아무런 알고리즘도 실행하지 않음

~~~Python
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
~~~

+ `session`을 생성하고 `run()`메소드를 `train`파라미터와 함께 호출

~~~Python
for step in xrange(8):
   sess.run(train)
print(step+1, sess.run(W), sess.run(b))
~~~

+ 입력 데이터에 최적화 된 직선의 `W`와 `b`를 찾기 위해 반복적인 프로세스 실행

> 최종결과

~~~Python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
vectors_set = []

for i in range(num_points):
         x1 = np.random.normal(0.0, 0.55)
         y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(8):
   sess.run(train)
print(step+1, sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
~~~

![results-3](/images/tensorflow-1/results-3.png)

## 반복에 따른 변화

~~~Python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
vectors_set = []

for i in range(num_points):
         x1 = np.random.normal(0.0, 0.55)
         y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for z in [1, 2, 3, 4, 5, 6, 7, 8, 1000]:
    for step in range(z):
        sess.run(train)
    print(step + 1, sess.run(W), sess.run(b))
    plt.plot(x_data, y_data, 'ro', label='Original data')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.legend()
    plt.show()
~~~

~~~Python
1 [-0.35690996] [0.30064374]
2 [-0.12226558] [0.3001456]
3 [0.02432493] [0.299835]
4 [0.08164252] [0.29971355]
5 [0.0964299] [0.2996822]
6 [0.09900617] [0.29967675]
7 [0.09931295] [0.2996761]
8 [0.09933809] [0.29967603]
1000 [0.09933957] [0.29967603]
~~~

![iter-1](/images/tensorflow-1/iter-1.png)
![iter-2](/images/tensorflow-1/iter-2.png)
![iter-3](/images/tensorflow-1/iter-3.png)
![iter-4](/images/tensorflow-1/iter-4.png)
![iter-5](/images/tensorflow-1/iter-5.png)
![iter-6](/images/tensorflow-1/iter-6.png)
![iter-7](/images/tensorflow-1/iter-7.png)
![iter-8](/images/tensorflow-1/iter-8.png)
![iter-1000](/images/tensorflow-1/iter-1000.png)
