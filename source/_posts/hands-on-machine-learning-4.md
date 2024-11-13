---
title: Hands-on Machine Learning (4)
date: 2022-08-11 15:33:19
categories:
- 5. Machine Learning
tags:
- Python
- TensorFlow
---
# Convolution Neural Network

+ 합성곱 신경망 (Convolution Neural Network, CNN)
  + 대뇌의 시각 피질 (cortex) 연구에서 시작
  + 이미지 검색, 자율주행, 영상 분류, 음성인식, 자연어 처리 등 다양한 분야에서 널리 사용

## The Architecture of the Visual Cortex

+ 뉴런들이 시야의 일부 범위 안에 있는 시각 자극에만 반응 (local receptive field)
+ 뉴런의 수용장들은 겹칠 수 있고 이를 합치면 전체 시야를 감싸게 됨
+ 동일한 수용장을 가지는 뉴런이여도 다른 각도의 선분에 반응하는 현상 발견
+ 특정 뉴런은 큰 수용장을 지니고 저수준 패턴이 조합된 상대적으로 복잡한 패턴에 반응

<!-- More -->

## Convolution Layer

+ 합성곱 층 (convolution layer)
  + 첫 번째 합성곱 층: 입력 이미지의 모든 픽셀이 연결되는 것이 아닌 합성곱 층 뉴런의 수용장 안에 있는 픽셀만을 연결
  + 두 번째 합성곱 층: 각 뉴런은 첫 번째 층의 작은 사각 영역 안에 위치한 뉴런에 연결
  + 은닉층: 더 큰 고수준 특성으로 조합하도록 원조
+ 스트라이드 (stride): 한 수용장과 다음 수용장 사이의 간격
+ 합성곱 커널 (convolution kernel): 뉴런의 가중치이며 필터를 거친 이미지는 특성 맵으로 변환
  + 특성 맵 (feature map): 필터를 가장 크게 활성화시키는 이미지의 영역 강조

~~~python
filters = np.zeros(shape = (7, 7, channels, 2), dtype = np.float32)

outputs = tf.nn.conv2d(images, filters, strides = 1, padding = "SAME")
conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = "relu")
~~~

+ `tf.nn.conv2d()`
  + `images`: 입력의 미니배치 (4D tensor)
    + 이미지: [높이, 너비, 채널]
    + 미니배치: [미니배치 크기, 높이, 너비, 채널]
  + `filters`: 적용될 필터 (4D tensor)
  + `strides`: 1 혹은 4개의 원소를 가지는 1D 배열
    + 4개의 원소: [배치 스트라이드, 수직 스트라이드, 수평 스트라이드, 채널 스트라이드]
  + `padding`
    + `VALID`: 합성곱 층에 제로 패딩 사용 X
    + `SAME`: 합성곱 층에 제로 패딩 사용
+ `tf.keras.layers.Conv2D`
  + 실제 CNN에서는 보통 훈련 가능한 변수로 필터를 정의하므로 해당 함수 사용

## Pooling Layer

+ 풀링 층 (pooling layer): 계산량과 메모리 사용량, 파라미터 수를 줄이기 위해 입력 이미지의 부표본 (subsample) 생성
  + 크기, 스트라이드, 패딩 유형 지정
  + 가중치가 존재하지 않음
  + 최대 혹은 평균과 같은 합산 함수를 통해 출력
+ 최대 풀링 (max pooling): 작은 변화에도 일정 수준의 불변성 (invariance) 보장
  + 입력값의 손실로 인해 파괴적인 단점 존재
+ 평균 풀링 (average pooling)
  + 일반적으로 최대 풀링 층의 성능이 상대적으로 높음
+ 전역 평균 풀링 (global average pooling): 각 특성 맵의 평균 연산
  + 각 샘플의 특성 맵마다 하나의 숫자 출력
  + 특성 맵에 있는 다양한 정보를 손실하지만 출력층에 유용

~~~python
max_pool = tf.keras.layers.MaxPool2D(pool_size = 2)
avg_pool = tf.keras.layers.AvgPool2D(pool_size = 2)
global_avg_pool = tf.keras.layers.GlobalAvgPool2D() # global_avg_pool = tf.keras.layers.Lambda(lambda X: tf.reduce_mean(X, axis = [1, 2]))
~~~

## CNN Architectures

~~~python
from tensorflow import keras
from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])
~~~

+ `input_shape = [28, 28, 1]` $\rightarrow$ 64개의 $7\times7$ 필터 (stride = 1)
+ `keras.layers.MaxPooling2D(pool_size=2)`: 풀링 크기가 2인 최대 풀링 층을 통해 공간 방향 차원을 절반으로 감소
+ 동일 구조 2회 반복
+ 출력층에 다다를수록 필터 개수를 늘려 저수준 특성을 연결하여 고수준 특성 연산 (풀링 층 다음에 필터 개수를 2배로 늘리는 것이 일반적)
  + 풀링 층이 공간 방향 차원을 절반으로 줄이므로 이어지는 층에서 파라미터의 개수, 메모리 사용량, 계산 비용을 크게 늘리지 않고 특성 맵 개수를 2배로 늘릴 수 있음
+ 밀집 네트워크를 통해 2개의 은닉층 및 1개의 출력층으로 구성된 완전 연결 네트워크 구성
  + `keras.layers.Flatten()`: 1D 배열을 위해 일렬로 펼침
  + `keras.layers.Dropout(0.5)`: 밀집 층 사이 과대적합 감소

***

# Recurrent Neural Network

+ 순환 신경망 (Recurrent Neural Network, RNN): 시계열 (time series) 데이터를 분석하여 미래 예측에 용이

## Recurrent Neuron and Recurrent Layer

+ 피드포워드 신경망과 유사하지만 뒤쪽으로 순환하는 연결의 차이점 존재
+ 각 타임 스텝 (time step) $t$마다 순환 뉴런 (recurrent neuron)은 $\boldsymbol{x}\_{(t)}$와 이전 타임 타임 스텝의 출력인 $y_{(t-1)}$을 입력받음

~~~python
model = keras.models.Sequential([
  keras.layers.SimpleRNN(1, input_shape = [None, 1])
])
~~~

## LSTM

+ 장단기 메모리 (Long Short-Term Memory, LSTM): 훈련이 빠르게 수렴하고 데이터에 존재하는 장기간 의존성 감지

~~~python
model = keras.models.Sequential([
  keras.layers.LSTM(20, return_sequence = True, input_shape = [None, 1]),
  keras.layers.LSTM(20, return_sequence = True),
  keras.layers.TimeDistributed(keras.layers.Dense(10))
])
~~~