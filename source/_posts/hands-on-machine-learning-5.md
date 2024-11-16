---
title: Hands-on Machine Learning (5)
date: 2022-08-15 16:19:00
categories:
- 5. Machine Learning
tags:
- Python
- TensorFlow
---
# Natural Language Processing

+ 자연어 처리 (Natural Language Processing, NLP): 컴퓨터와 사람의 언어 사이의 상호작용에 대해 연구하는 컴퓨터 과학과 어학의 한 분야
  + 문자 단위 RNN (character RNN): 문장에서 다음 글자를 예측하도록 훈련

~~~python 
from tensorflow.compat.v2 import keras

shakespeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level = True)
tokenizer.fit_on_texts(shakespeare_text)
~~~

+ 셰익스피어 작품을 다운로드 이후 `Tokenizer`를 통해 모든 글자를 정수로 인코딩

<!-- More -->

> Encoding and decoding
<img src="/images/hands-on-machine-learning-5/encoding-and-decoding.png" alt="encoding-and-decoding" width="480" />

~~~python
import numpy as np

max_id = len(tokenizer.word_index)
dataset_size = tokenizer.document_count
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
~~~

+ 순차 데이터셋
  + 처음 90%: 훈련 세트
  + 이후 5%: 검증 세트
  + 이후 5%: 태스트 세트
+ RNN: 암묵적으로 과거 (훈련 세트)에서 학습하는 패턴이 미래에 등장한다고 가정

~~~python
n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift = 1, drop_remainder = True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth = max_id), Y_batch))
dataset = dataset.prefetch(1)
~~~

+ TBPTT (truncated backpropagation through time): `window()` 메서드를 이용해 긴 시퀀스를 작고 많은 윈도우로 변환
  + `shift = 1` 가장 큰 훈련 세트 제작 가능
+ 중첩 데이터셋 (nested dataset): 각각 하나의 데이터셋으로 표현되는 윈도우를 포함하는 데이터셋 제작
+ 플랫 데이터셋 (flat dataset): 중첩 데이터셋을 텐서로 변환 $\rightarrow$ `flat_map()`
+ 윈도우를 섞고 배치로 변환하여 입력과 타깃 분리 후 원-핫 벡터를 사용하여 인코딩

~~~python
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences = True, input_shape = [None, max_id], dropout = 0.2, recurrent_dropout = 0.2),
    keras.layers.GRU(128, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation = "softmax"))
])

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam")
history = model.fit(dataset, epochs = 20)
~~~

+ 이전 글자 100개를 기반으로 다음 글자를 예측하기 위해 유닛 128개의 `GRU` 층 2개와 입력 (dropout)과 은닉 상태 (recurrent_dropout)에 20% 드롭아웃 사용
+ 텍스트의 고유 글자 수인 39개 (`max_id`)로 출력층의 유닛 구성

***

# Autoencoder & Generative Adversarial Networks

+ 생성 모델 (generative model): 오토인코더를 통해 훈련 데이터와 비슷한 새로운 데이터 생성
+ 생산적 적대 신경망 (Generative Adversarial Networks, GAN): 초해상도, 이미지 컬러화, 이미지 편집, 동영상 프레임 예측, 데이터 증식 등 타 모델의 취약점을 식별하고 개선
  + 생성자 (generator): 훈련 데이터와 비슷하게 보이는 데이터 생성
  + 판별자 (discriminator): 가짜 데이터와 진짜 데이터 분류
  + 적대적 훈련 (adversarial training): 경쟁하며 신경망을 훈련
+ 오토인코더 (autoencoder)
  + 인코더 (encoder), 인지 네트워크 (recognition network): 입력을 내부 표현으로 전환
  + 디코더 (decoder), 생성 네트워크 (generative network): 내부 표현을 출력으로 전환
  + 출력: 재구성 (reconstruction)
  + 비용 함수: 재구성이 입력과 다를 때 모델에 벌점을 부과하는 재구성 손실 (reconstruction loss)

## Stacked Autoencoder

+ 적층 오토인코더 (stacked autoencoder), 심층 오토인코더 (deep autoencoder): 오토인코더가 은닉층을 가지는 경우

~~~python
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

tf.random.set_seed(42)
np.random.seed(42)

stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",
                   optimizer=keras.optimizers.SGD(learning_rate=1.5), metrics=[rounded_accuracy])
history = stacked_ae.fit(X_train, X_train, epochs=20,
                         validation_data=(X_valid, X_valid))
~~~

+ `stacked_encoder`
  + $28\times28$ 픽셀의 흑백 이미지 입력
  + 784 크기의 벡터로 `Flatten()`
  + `Dense()`를 이용해 hidden layer 구성
  + 30 크기의 벡터 출력
+ `stacked_decoder`
  + 30 크기의 벡터 입력
  + `Dense()`를 이용해 hidden layer 구성
  + $28\times28$ 픽셀의 흑백 이미지로 `Reshape()`
+ `BCELoss` > `MSELoss`: 적층 오토인코더를 컴파일하는 경우 적합

> Original images and reconstructed images
<img src="/images/hands-on-machine-learning-5/original-images-and-reconstructed-images.png" alt="original-images-and-reconstructed-images" width="1014" />

> Result of t-SNE
![result-of-t-sne](/images/hands-on-machine-learning-5/result-of-t-sne.png)

## Generative Adversarial Networks

~~~python
np.random.seed(42)
tf.random.set_seed(42)

codings_size = 30

generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
        plot_multiple_images(generated_images, 8)
        plt.show()
        
train_gan(gan, dataset, batch_size, codings_size, n_epochs=1)
~~~

[GAN Training Process](https://zerohertz.github.io/generative-adversarial-network-4/#Training-and-Results)

+ 내시 균형 (Nash equillibrium): 다른 플레이어가 전략을 수정하지 않을 것이므로 어떤 플레이어도 자신의 전략을 수정하지 않는 상태
  + GAN에서는 생성자가 완벽히 실제와 생성 이미지를 분류한 경우 의미
+ 모드 붕괴 (mode collapse): 생성자의 출력 다양성 감소

### Deep Convolutional GAN

~~~python
tf.random.set_seed(42)
np.random.seed(42)

codings_size = 100

generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                 activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME",
                                 activation="tanh"),
])
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])
~~~

+ 생성자와 판별자에 convolution layer를 추가하여 이미지 생성 용이

### StyleGAN

+ StyleGAN: 생성자에 style transfer 기법을 사용해 생성된 이미지가 같은 다양한 크기의 국부적인 구조를 가지도록 구성
+ 매핑 네트워크: 코딩을 여러 스타일 벡터로 매핑
  + 아핀 변환 (affine transformation): 잠재 표현 (코딩) $z$를 벡터 $w$로 매핑
+ 합성 네트워크
  + 입력과 모든 합성곱 층의 출력에 약간의 잡음 추가
  + 잡음이 섞인 이후 적응적 인스턴스 정규화 (Adaptive Instance Normalization, AdaIN) 추가