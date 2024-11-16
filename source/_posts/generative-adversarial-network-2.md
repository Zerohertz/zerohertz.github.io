---
title: Generative Adversarial Network (2)
date: 2022-07-20 14:59:06
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# GAN (Generative Adversarial Network)

## Image Generation

+ Traditional neural network: 정보를 감소, 정제, 축약
  + Ex) $28\times28\rightarrow784\ (input)\rightarrow10\rightarrow1\ (output)$
+ Backquery: 기존의 신경망을 반대로 뒤집어 레이블을 통해 이미지 생성 (원핫 인코딩 벡터를 훈련된 네트워크에 넣어 레이블에 맞는 이상적 이미지 생성)
  + Ex) $1\ (input)\rightarrow10\rightarrow784\ (output)\rightarrow28\times28$
  + 같은 원핫 인코딩 벡터인 경우 같은 결과 출력
  + 각 레이블을 나타내는 모든 훈련 데이터의 평균적 이미지 도출 $\rightarrow$ 한계점 (훈련 샘플로 사용 불가)

<!-- More -->

## Adversarial Training

> 생산적 적대 신경망 (Generative Adversarial Network, GAN)의 기본 개념

+ Generator (생성기): 허구의 이미지를 생성하는 신경망
  + 판별기를 속이는 경우 보상
  + 판별기에게 적발될 경우 벌
+ Discriminator (판별기): 실제 이미지와 허구 이미지를 분류하는 신경망
  + 생성기를 통해 생성된 이미지를 허구 이미지로 분류한 경우 보상
  + 생성기를 통해 생성된 이미지를 실제 이미지로 분류한 경우 벌

## GAN Training

1. 실제 데이터를 판별기가 `1`로 분류할 수 있도록 판별기 업데이트
2. 생성기를 통해 생성된 데이터를 `0`으로 분류할 수 있도록 판별기만을 업데이트 (생성기 업데이트 X)
3. 판별기가 생성기를 통해 생성된 데이터를 `1`로 분류하도록 생성기만을 업데이트 (판별기 업데이트 X)

+ 생성기와 판별기가 서로 적대적인 (두 모델의 성능이 비슷) 경우 적절한 훈련 가능
+ 생성기 혹은 판별기 중 한 모델만 성능이 개선될 경우 최종 성능의 큰 하락 발생

***

# Simple Pattern

## Real Data Source

> Real data generation function
![Real data generation function](/images/generative-adversarial-network-2/180653460-44f175ba-fe22-4432-b7a7-ec68feaf8193.png)

## Discriminator

> Discriminator
~~~python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 3),
            nn.Sigmoid(),
            nn.LayerNorm(3),
#             nn.LeakyReLU(0.02),
            nn.Linear(3, 3),
            nn.Sigmoid(),
#             nn.LayerNorm(3),
#             nn.Linear(3, 3),
#             nn.Sigmoid(),
            nn.LayerNorm(3),
            nn.Linear(3, 1),
            nn.Sigmoid()
#             nn.LeakyReLU(0.02)
        )
#         self.loss_function = nn.MSELoss()
        self.loss_function = nn.BCELoss()
#         self.optimiser = torch.optim.SGD(self.parameters(), lr = 0.005)
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.001)
        self.counter = 0
        self.progress = []
        pass
    
    def forward(self, inputs):
        return self.model(inputs)
    
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
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns = ['loss'])
        df.plot(ylim = (0, 1.0), figsize = (16, 8), alpha = 0.1, marker = '.', grid = True, yticks = (0, 0.25, 0.5))
        pass
~~~

> Training discriminator
![Training discriminator](/images/generative-adversarial-network-2/180654877-d226b1b0-333e-4828-a457-d32eaa3da51d.png)

## Generator

> Generator

~~~python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(3),
            nn.Linear(3, 3),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(3),
            nn.Linear(3, 3),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(3),
            nn.Linear(3, 6),
            nn.LeakyReLU(0.02)
        )
#         self.optimiser = torch.optim.SGD(self.parameters(), lr = 0.01)
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.01)
        self.counter = 0
        self.progress = []
        pass
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)
        loss = D.loss_function(d_output, targets)
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns = ['loss'])
        df.plot(ylim = (0, 1.0), figsize = (16, 8), alpha = 0.1, marker = '.', grid = True, yticks = (0, 0.25, 0.5))
        pass
~~~

+ `self.loss`: 생성기는 판별기로부터 입력된 기울기 오차를 통해 업데이트되므로 생성기의 손실 함수는 정의되지 않음
+ `self.train()`: 생성기 훈련 시 판별기의 결과로 계산된 손실의 역전파 값 필요
  1. 입력값 $\rightarrow$ `self.forward(inputs)` $\rightarrow$ `g_output`
  2. `D.forward(g_output)` $\rightarrow$ `d_output`
  + 손실은 `d_output`과 목푯값 간의 차이로 산출
  + 손실로부터 오차 역전파 $\rightarrow$ `self.optimiser` (`D.optimiser` X)

## GAN Training

> GAN Training

~~~python
D = Discriminator()
G = Generator()

image_list = []

for i in range(100000):
    D.train(generate_real(), torch.FloatTensor([1.0]))
    D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
    G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
    if (i % 100 == 0):
        image_list.append(G.forward(torch.FloatTensor([0.5])).detach().numpy())
    pass
~~~

1. 판별기 및 생성기의 객체 생성
2. 실제 데이터에 대해 판별기 훈련
3. 생성기에서 생성된 데이터를 판별기에 훈련
   + `detach()`: 생성기의 출력에 적용되어 계산에서 생성기 분리
     + 생성기의 출력에 적용
     + 효율적이고 빠른 결과 도출을 위해 사용
   + `backwards()`: 판별기의 손실에서 기울기 오차를 계산의 전 과정에 걸쳐 계산
     + 판별기의 손실, 판별기, 생성기까지 모두 전해짐
     + 이 단계에서는 판별기를 훈련하는 것이므로 생성기의 기울기를 계산할 필요 X
4. 생성기 훈련 및 입력값을 0.5로 설정하여 판별기 객체에 전달
   + `detach()` X: 오차가 판별기로부터 생성기까지 전달돼야함


> Loss of discriminator and generator
![Loss of discriminator and generator](/images/generative-adversarial-network-2/180659175-5a817c42-4154-4bae-818f-90d4a43b8d84.png)

+ 판별기와 생성기의 손실: 0.69에 수렴
  + 이진 교차 엔트로피 (`BCELoss()`)에서의 $ln(2)$
    + 판별기가 실제 데이터와 생성된 데이터를 잘 판별하지 못함
    + 생성기가 판별기를 속일 수 있는 성능
  + 평균제곱오차 (`MSELoss()`): 0.25
    + 판별기가 실제 이미지와 생성 이미지를 잘 판별하지 못한 경우 $\rightarrow$ 출력: `0.5`
    + $0.5^2 = 0.25$

> Pattern of generator as training progresses
![Pattern of generator as training progresses](/images/generative-adversarial-network-2/180659184-9694252e-46f3-4543-a80d-2b91821ed258.png)

