---
title: Generative Adversarial Network (1)
date: 2022-07-19 18:51:06
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# PyTorch

## Setup

[How to install PyTorch in M1 Mac](https://velog.io/@hwanseung2/M1-Mac%EC%97%90%EC%84%9C-Pytorch-GPU-%EA%B0%80%EC%86%8D)

~~~python
conda install pytorch torchvision torchaudio -c pytorch-nightly
~~~

![RIP Kernel](https://user-images.githubusercontent.com/42334717/179754313-48642bc4-affd-45c0-a9c2-dc2b53c1439a.png)

+ `conda activate env` 이후 `jupyter notebook`을 통해 `import torch`를 실행할 경우 위의 사진과 같이 커널이 죽는다.
+ 따라서 아래 명령어를 통해 `jupyter notebook`에서 커널을 선택할 수 있게 해줘야한다.

~~~python
conda install nb_conda_kernels
~~~

![Change kernel](https://user-images.githubusercontent.com/42334717/179777082-69e27242-9c36-4491-815a-30cbc3a4ba2c.png)

+ 위의 모듈 `nb_conda_kernels`를 통해 모든 가상환경들을 선택해서 `jupyter notebook`에서 사용할 수 있다.

<!-- More -->

## Tensor and Gradient

~~~python In
x = torch.tensor(3.5)
y = x + 2
print(x,y)
~~~

~~~python Out
tensor(3.5000) tensor(5.5000)
~~~

+ 일반적 계산과 다르게 `PyTorch` 내에서의 수식은 $y(x) = x$와 같이 저장된다.

~~~python In
x = torch.tensor(4., requires_grad = True)
y = pow(x, 2) + 1
print(y) #함수값 출력
y.backward()
print(x.grad) #미분값 출력
~~~

~~~python Out
tensor(17., grad_fn=<AddBackward0>)
tensor(8.)
~~~

+ 이와 같이 $y(x) = x$로 관계가 저장되기 때문에 $y'(x)$를 산정할 수 있다.

~~~python In
x = torch.tensor(4., requires_grad = True)
y = x*x
z = 3*y + 1 # 3x^2 + 1
z.backward()
print(x.grad)
~~~

~~~python Out
tensor(24.)
~~~

+ 매개변수와 같은 응용도 가능하다.
+ [이 외의 tensor handling](https://zerohertz.github.io/neural-network-example/)

***

# Neural Network based on PyTorch

## Data Description (MNIST)

[Download MNIST Train Data](https://pjreddie.com/media/files/mnist_train.csv)
[Download MNIST Test Data](https://pjreddie.com/media/files/mnist_test.csv)

> Read MNIST Data
![MNIST](https://user-images.githubusercontent.com/42334717/179787188-9956c4ac-d0c3-4550-8aa6-b9d54145458f.png)

+ 첫 숫자는 해당 이미지의 label을 의미한다.
+ 나머지 784개의 숫자는 $28\times 28$으로 이뤄진 이미지의 각 픽셀 값이다.

> Data Visualization
![Data Visualization](https://user-images.githubusercontent.com/42334717/179788444-07444eab-f2fa-4488-8d46-8226aa1cb35c.png)

## Artificial Neural Network

+ 앞서 말한 것과 같이 이미지는 $28\times 28$ 즉, 784개의 픽셀 값으로 이루어져있다.
+ 따라서 input layer는 784개의 node를 지녀야한다.
+ 또한 label의 종류가 10개 (0 ~ 9) 이므로 output layer는 10개의 node를 지녀야한다.
+ 가장 간단한 신경망을 구성하기 위해 아래 항들을 따른다.
  + 특정 layer의 모든 node들은 그 다음 레이어의 모든 node와 연결 (fully connected)
  + Input layer와 output layer 사이에 존재하는 hidden layer의 크기는 200
  + Hidden layer와 output layer 사이의 activation function은 logistic function


~~~python
import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []
        pass
    
    def forward(self, inputs):
        return self.model(inputs)
~~~

|문법|명칭|의미|
|:-:|:-:|:-:|
|`ANN(nn.Module)`|클래스 이름|nn.Module로부터 상속|
|`__init__()`|생성자 (constructor)|-|
|`super.__init__()`|-|부모 클래스의 생성자 호출|
|`nn.Sequential()`|-|파라미터를 통해 간단한 레이어 정의|
|`nn.Linear(m,n)`|-|m개의 노드로부터 n개의 노드까지의 선형 완전 연결 매핑|
|`nn.Sigmoid()`|-|로지스틱 활성화 함수를 이전 레이어의 출력에 적용|
|`nn.MSELoss()`|-|신경망에서 오차를 정의하는 방법 중 하나|
|`torch.optim.SGD()`|-|손실을 토대로 신경망의 가중치를 수정하는 방법 중 하나|

+ `nn.Linear()`: $Ax+B$와 같은 형태로 노드 사이를 연결
  + $A$: 가중치
  + $B$: 편향 (bias)
  + 위 두 파라미터를 학습 파라미터 (learnable parameter)라고 명함
+ `nn.MSELoss()`: 평균제곱오차 (Mean Squared Error)을 통해 실제와 예측된 결과 사이의 차이를 제곱하고 평균내어 계산
  + Loss function: 학습 파라미터 업데이트하기 위해 오차 계산
  + Error (오차) vs. Loss (손실)
    + Error: 정답과 예측값 사이의 차이
    + Loss: 궁극적으로 풀어야 할 문제에 대한 오차
    + 비슷하지만 딥러닝에서 굳이 따지자면 **Loss**를 기반으로 신경망의 가중치 업데이트
+ `torch.optim.SGD()`: 확률적 경사 하강법 (Stochastic Gradient Descent)
  + `self.parameters()`를 통해 세세한 설정 가능
+ `ANN.forward()`
  + 입력값을 받아 `nn.Sequential()`에서 정의한 `self.model()`에 전달
  + 모델의 결과는 `forward()`를 호출한 곳으로 전달

~~~python
...
    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
~~~

+ `ANN.train()`: 구성한 신경망의 훈련을 위한 메서드
  + 신경망에 전달할 입력과 원하는 목표의 출력으로 구성 $\rightarrow$ 손실 계산
  1. `self.forward(inputs)` $\rightarrow$ `self.model(inputs)` $\rightarrow$ `outputs`
     + 입력값을 신경망에 전달하여 결과 산출
  2. `outputs` $\rightarrow$ `self.loss_function()` $\rightarrow$ `loss`
     + 신경망의 손실 계산
  3. `self.optimiser.zero_grad()` $\rightarrow$ `loss` $\rightarrow$ `loss.backward()` $\rightarrow$ `self.optimiser.step()`
     + 계산 그래프의 기울기 초기화 후 손실을 통해 신경망의 가중치 업데이트
     + 기울기 초기화 제외 시 `loss.backward()`를 따라 모든 계산에 중첩

## Training Process Visualization

~~~python
...
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
~~~

+ 훈련이 진행되는 동안 매 10개의 훈련 샘플마다 손실값을 저장하여 시각화

## Data Handling for PyTorch

+ `torch.utils.data.Dataset` 객체 이용

~~~python
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        pass
    
    pass
~~~

|문법|의미|
|:-:|:-:|
|`__len__()`|데이터셋의 길이 반환|
|`__getitem__()`|데이터셋의 n번째 아이템 반환|

> `__getitem__()`
<img width="497" alt="getitem" src="https://user-images.githubusercontent.com/42334717/179807111-db80c2f6-f1fd-4cc0-9448-371515ff3184.png">

> `plot_image()`
![plot_image](https://user-images.githubusercontent.com/42334717/179806567-facb03df-ef86-4d79-8a8c-0a7549bc3968.png)


## Classifier Training

> Classifier training
![Classifier training](https://user-images.githubusercontent.com/42334717/179815267-744dfab0-a705-43c1-a2b6-28f89947a777.png)

> Plot loss chart
![plot loss chart](https://user-images.githubusercontent.com/42334717/179815343-b6e690df-7b43-46d6-8676-c892ee0e35c8.png)

## Classifier Validation

> Classification of test data
![Classification of test data](https://user-images.githubusercontent.com/42334717/179815843-51c1f128-4e0d-41ff-b20c-0df0be866aea.png)

> Classifier validation
![Classifier validation](https://user-images.githubusercontent.com/42334717/179816249-a38d4d7c-98ce-4abd-97ad-f1229abe27fc.png)

+ 87.99%의 분류 정확도

## GPU 가속

~~~python
import torch
import torch.nn as nn

class ANN_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []
        pass
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def train(self, inputs, targets, device):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
    
    pass

%%time

device = torch.device("mps")
C_GPU = ANN_GPU()
C_GPU = C_GPU.to(device)
LearningData = MnistDataset('MNIST/mnist_train.csv')

epochs = 4

for i in range(epochs):
    print('training epoch', i+1, "of", epochs)
    for label, image_data_tensor, target_tensor in LearningData:
        C.train(image_data_tensor, target_tensor, device)
        pass
    pass
~~~

+ 근데 CPU로 훈련하는게 더 빠름...

***

# Neural Network Reinforcement

## Loss Function

~~~python
...
        self.loss_function = nn.BCELoss()
...
~~~

> Reinforcing Neural Network by Changing Loss Function
<img width="1021" alt="Reinforcing Neural Network by Changing Loss Function" src="https://user-images.githubusercontent.com/42334717/179886806-c3a5bd36-4fdb-411a-95ac-5abf2f3f5217.png">

+ 이진 교차 엔트로피 (Binary Cross Entropy, BCE) 손실: Classification에서 loss function으로 자주 사용
  + 확실하게 틀린 경우 큰 페널티 부여
  + `MSELoss()`에 비해 반복에 따라 손실이 느리게 감소
  + `BCELoss()`을 사용함으로 `MSELoss()`를 사용한 모델에 비해 87.99%에서 91.02%으로 분류 정확도 향상

## Activation Function

~~~python
...
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 10),
            nn.LeakyReLU(0.02)
        )
        self.loss_function = nn.MSELoss()
...
~~~

> Reinforcing Neural Network by Changing Activation Function
<img width="1021" alt="Reinforcing Neural Network by Changing Activation Function
" src="https://user-images.githubusercontent.com/42334717/179895206-993af8af-4586-475c-a1b2-fdaab54f3f49.png">

+ Logistic function: 뉴런에서 일어나는 신호 전달 현상과 비슷하여 초기의 신경망에서 자주 사용
  + 수학적으로 기울기를 도출하기 간단
  + 큰 값들에 대해 기울기가 작고 사라질 수 있음
  + 소실될 경우 이를 포화 (saturation)이라고 함
+ 정류 선형 유닛 (Rectified Linear Unit, ReLU)
  + 0보다 큰 값들에 대해 일정한 기울기
  + 0보다 작은 값들에 대해 경사가 0이기 때문에 기울기가 소실되는 문제가 여전히 존재
+ Leaky ReLU
  + 0보다 작은 경우 미세한 기울기 허용
  + 손실 함수로 `BCELoss()` 사용 불가 $\rightarrow$ BCE 손실은 0과 1 사이 외의 값을 받을 수 없음
  + 91.02%에서 97.07%으로 분류 정확도 향상

## Optimizer

~~~python
...
        self.optimiser = torch.optim.Adam(self.parameters())
...
~~~

> Reinforcing Neural Network by Changing Optimizer
<img width="1021" alt="Reinforcing Neural Network by Changing Optimizer" src="https://user-images.githubusercontent.com/42334717/179896344-a72cca9f-b0eb-48f5-975f-c08fa8a80e2d.png">

+ 확률적 경사 하강법 (Stochastic Gradient Descent, SGD)
  + 국소 최적해에 빠질 가능성 존재
  + 모든 학습 파라미터에 단일한 학습률 적용
+ Adam
  + 관성을 이용하여 국소 최적해로 빠질 가능성 최소화
  + 각 학습 파라미터에 대해 다른 학습률 적용

## Normalization

~~~python
...
self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 10),
            nn.LeakyReLU(0.02)
        )
...
~~~

> Reinforcing Neural Network by Normalization
<img width="1021" alt="Reinforcing Neural Network by Normalization" src="https://user-images.githubusercontent.com/42334717/179897536-fc204ebf-d1c2-4d3d-bb3b-f8c671fc7e13.png">

+ 신경망의 가중치 혹은 신호의 값에 대해 peak로 인해 중요한 값이 소실될 수 있음
+ 따라서 파라미터들의 범위를 조절하거나 평균을 0으로 설정하는 방법 사용 $\rightarrow$ 정규화 (normalization)

## Combination

~~~python
...
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters())
        self.counter = 0
        self.progress = []
        pass
...
~~~

> Reinforcing Neural Network
<img width="1021" alt="Reinforcing Neural Network" src="https://user-images.githubusercontent.com/42334717/179899497-c78b9eba-6021-4362-b64a-ed01b1c47a69.png">

+ Loss Function
+ Activation Function
+ Optimizer
+ Normalization

***

# CUDA

+ Tensor operation speed: Vanila python <<< Numpy
+ CUDA (Compute Unified Device Architecture): GPU (Graphic Processing Unit) 기반 머신러닝 표준 소프트웨어 프레임워크

~~~python
device = torch.device("mps") #CUDA 아님. . .
~~~

> CUDA는 아니지만 M1 Mac에서의 GPU 가속...
<img width="644" alt="GPU" src="https://user-images.githubusercontent.com/42334717/179909787-04998599-3d3b-466c-8634-fa61a911abe4.png">