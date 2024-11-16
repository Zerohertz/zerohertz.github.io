---
title: Generative Adversarial Network (3)
date: 2022-07-26 23:17:37
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# MNIST Dataset

## Data Description

> MNIST dataset
![mnist-dataset](/images/generative-adversarial-network-3/mnist-dataset.png)

+ Dataset 구성: 0 ~ 9의 손글씨 이미지 ($28\times28=784$)와 label
+ `MnistDataset` 클래스
  + `Raw data` $\rightarrow$ `Tensor`
  + `Label`, `Pixel values`, `One-hot encoding tensor` 반환
+ 목표: 생성기의 생성 이미지가 판별기를 속일 수 있도록 훈련

<!-- More -->

## Discriminator

~~~python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
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
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
    pass
~~~

> Discriminator training
![discriminator-training](/images/generative-adversarial-network-3/discriminator-training.png)

+ 실제 이미지 vs. 임의의 노이즈 분류 확인

## Generator

~~~python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.optimiser = torch.optim.SGD(self.parameters(), lr = 0.01)
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
    pass
~~~

+ 생성기는 훈련 데이터의 여러 양상을 다양하게 반영해야 함
  + 항상 정확히 같은 값 출력 X
  + 0 ~ 9의 모든 이미지 생성
+ 하지만 신경망의 같은 입력값에 대해서 항상 같은 출력
  + 생성기의 입력으로 상수 X
  + 매 훈련 사이클마다 임의 입력 필요 $\rightarrow$ Random Seed

## GAN Training

~~~python CPUTraining.py
%%time

D = Discriminator()
G = Generator()

for label, image_data_tensor, target_tensor in mnist_dataset:
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    D.train(G.forward(generate_random(1)).detach(), torch.FloatTensor([0.0]))
    G.train(D, generate_random(1), torch.FloatTensor([1.0]))
    pass
~~~

~~~python GPUAcceleratedTraining.py
%%time

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())

D = Discriminator()
D = D.to("mps")
G = Generator()
G = G.to("mps")

label, image_data_tensor, target_tensor = mnist_dataset[:]

image_data_tensor_g = image_data_tensor.to("mps")

tFT1 = torch.FloatTensor([1.0]).to("mps")
tFT0 = torch.FloatTensor([0.0]).to("mps")

Dt = generate_random(len(mnist_dataset)).to("mps")
Gt = generate_random(len(mnist_dataset)).to("mps")
Dt = Dt.reshape(len(Dt),1)
Gt = Dt.reshape(len(Gt),1)

for image_data_tensor, dt, gt in zip(image_data_tensor_g, Dt, Gt):
    D.train(image_data_tensor, tFT1)
    D.train(G.forward(dt).detach(), tFT0)
    G.train(D, gt, tFT1)
    pass
~~~

> Loss of GAN training process
![loss-of-gan-training-process-1](/images/generative-adversarial-network-3/loss-of-gan-training-process-1.png)

+ Discriminator
  + Loss: 0 $\rightarrow$ 0.25 $\rightarrow$ 0
  + 판별기 > 생성기 $\rightarrow$ 판별기 = 생성기 $\rightarrow$ 판별기 > 생성기
+ Generator
  + Loss: 1 $\rightarrow$ 0.25 $\rightarrow$ 1
  + 판별기 > 생성기 $\rightarrow$ 판별기 = 생성기 $\rightarrow$ 판별기 > 생성기

> Results of generator
![results-of-generator-1](/images/generative-adversarial-network-3/results-of-generator-1.png)

+ 기존의 숫자 이미지들과 형상 유사
+ 생성기 이미지들의 차이가 거의 없음

## Mode Collapse

+ Definition: 생성기가 다양한 label에 대한 이미지를 생성하지 못하고 하나 또는 극히 일부의 이미지만 생성
  + 발생 이유는 명확히 규명되지 않음
  + 생성기가 판별기보다 더 앞서간 후 항상 실제에 가깝게 결과가 나오는 지점에 대해서만 이미지를 만들어내는 가능성 존재
+ 손실이 높아지는 구간에서는 학습이 진행되지 않음 (생성기의 성능 향상 불가)
  + 훈련의 질 중요

~~~python AdvancedGAN.py
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02), #nn.Sigmoid(),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss() #nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001) #torch.optim.SGD(self.parameters(), lr=0.01)
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
        df.plot(ylim = (0), figsize = (16, 8), alpha = 0.1, marker = '.', grid = True, yticks = (0, 0.25, 0.5, 1.0, 5.0))
        pass
    pass

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02), #nn.Sigmoid(),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001) #torch.optim.SGD(self.parameters(), lr = 0.01)
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
        df.plot(ylim = (0), figsize = (16, 8), alpha = 0.1, marker = '.', grid = True, yticks = (0, 0.25, 0.5, 1.0, 5.0))
        pass
    pass

D = Discriminator()
G = Generator()

for label, image_data_tensor, target_tensor in mnist_dataset:
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    D.train(G.forward(generate_random(100)).detach(), torch.FloatTensor([0.0]))
    G.train(D, generate_random(100), torch.FloatTensor([1.0]))
    pass
~~~

> Results of advanced generator: edit activation function, layer, optimizer
![results-of-advanced-generator-1](/images/generative-adversarial-network-3/results-of-advanced-generator-1.png)

+ 생성기의 이미지가 조금 더 선명해지고 서로 식별이 약간 가능해졌지만 숫자의 형태라고 보기엔 어려움
+ 하나의 시드를 통해 10개 숫자에 대한 784개의 픽셀을 생성하는 것을 시도하기 때문에 개선 필요

~~~python
...
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02), #nn.Sigmoid(),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
...
for label, image_data_tensor, target_tensor in mnist_dataset:
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    D.train(G.forward(generate_random(100)).detach(), torch.FloatTensor([0.0]))
    G.train(D, generate_random(100), torch.FloatTensor([1.0]))
    pass
~~~

> Results of advanced generator: edit random seed size
![results-of-advanced-generator-2](/images/generative-adversarial-network-3/results-of-advanced-generator-2.png)

+ 여전히 모드 붕괴 발생
+ 판별기에 입력되는 임의의 이미지 픽셀 값은 0에서 1 사이에서 고르게 선택
  + 0 ~ 1: 실제 데이터셋 기반 값
  + 정규분포와 같은 경향성 없이 선택
+ 생성기 투입 시드: 평균이 0이고 분산이 1인 분포에서 선택
  + 신경망에서 평균이 0이고 분산이 제한된 정규화된 값들이 학습에 유리

~~~python genRandom.py
def generate_random_image(size):
    random_data = torch.rand(size)
    return(random_data)

def generate_random_seed(size):
    random_data = torch.randn(size)
    return(random_data)
~~~

> Results of advanced generator: edit random seed
![results-of-advanced-generator-3](/images/generative-adversarial-network-3/results-of-advanced-generator-3.png)

+ 모드 붕괴가 해결되어 다른 종류의 숫자 이미지를 생성기가 생성 가능

> Loss of advanced GAN training process
![loss-of-advanced-gan-training-process](/images/generative-adversarial-network-3/loss-of-advanced-gan-training-process.png)

+ `BCELoss()`: 이진 교차 엔트로피의 수학적 정의에 의해 $\ln(2)=0.693$이 이상적 손실
+ 모드 붕괴의 해결책이 무조건 `randn()`의 사용 혹은 시드의 수를 증가시키는 것은 아님
  + GAN에서 생성기와 판별기의 균형을 맞추는 것은 상당히 어려운 과정
  + 균형이 맞지 않더라도 생성기의 성능이 나쁘지 않을 수 있음

## Random Seed

+ 두가지 seed를 이용해 그 사이에 존재하는 seed들에 의해 생성기에 의해 생성된 이미지 실험

> Results of advanced generator: edit epoch
![results-of-advanced-generator-4](/images/generative-adversarial-network-3/results-of-advanced-generator-4.png)

> Images generated by seed1 and seed2
![images-generated-by-seed1-and-seed2](/images/generative-adversarial-network-3/images-generated-by-seed1-and-seed2.png)

> Images generated by seed1 + seed2 and seed1 - seed2
![images-generated-by-seed1+seed2-and-seed1-seed2](/images/generative-adversarial-network-3/images-generated-by-seed1+seed2-and-seed1-seed2.png)

> Images generated by seeds between seed1 and seed2
![images-generated-by-seeds-between-seed1-and-seed2](/images/generative-adversarial-network-3/images-generated-by-seeds-between-seed1-and-seed2.png)

***

# Face Image

+ 3차원 (RGB) 풀컬러 이미지 훈련 및 생성
  + $M\times N\times3$
+ 사진의 훈련 데이터셋을 이용 및 다양하고 실제적 결과 산출

## CelebA Dataset

+ 202,599개의 유명인 얼굴 이미지
+ 눈과 입의 위치가 비슷한 좌표에 위치하도록 조정됨 (aligned)

[CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Hierarchical Data Format

+ 다수의 `.jpeg` 파일 $\rightarrow$ 훈련 과정에서 열고 닫을 시 시간 소모 큼 $\rightarrow$ HDF5 사용
+ HDF (Hierarchical Data Format, 계층적 데이터 형식): 용량이 매우 큰 데이터에 효과적으로 접근하기 위해 만들어진 데이터 형식
  + 하나 이상의 그룹을 가질 수 있어 계층적이라 불림
  + 그룹 안에 여러 개의 데이터셋이 포함될 수 있음
  + HDF5 (HDF version 5)를 이용해 훈련 과정에서의 시간 소모 개선 및 메모리의 한계 극복

~~~python
import torchvision.datasets

CelebA_dataset = torchvision.datasets.CelebA(root='.', download=True)

import h5py
import zipfile
import imageio
import os

hdf5_file = 'celeba/celeba_aligned_small.h5py'

total_images = 20000

with h5py.File(hdf5_file, 'w') as hf:
    count = 0
    with zipfile.ZipFile('celeba/img_align_celeba.zip', 'r') as zf:
        for i in zf.namelist():
            if (i[-4:] == '.jpg'):
                ofile = zf.extract(i)
                img = imageio.imread(ofile)
                os.remove(ofile)
                hf.create_dataset('img_align_celeba/'+str(count)+'.jpg', data=img, compression="gzip", compression_opts=9)
                count = count + 1
                if (count%1000 == 0):
                    print("images done .. ", count)
                    pass
                if (count == total_images):
                    break
                pass
            pass
        pass
~~~

+ `import h5py`: HDF5 파일을 다루기 위한 라이브러리
+ $218\times178\times3$: 높이 218픽셀, 너비 178픽셀, RGB

~~~python
class CelebADataset(Dataset):
    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (index >= len(self.dataset)):
            raise IndexError()
        img = np.array(self.dataset[str(index)+'.jpg'])
        return torch.cuda.FloatTensor(img) / 255.0

    def plot_image(self, index):
        plt.imshow(np.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')
        pass
    pass

celeba_dataset = CelebADataset('celeba/celeba_aligned_small.h5py')
~~~

+ `__init__()`: HDF5 파일을 열고 `img_align_celeba`로 각각의 이미지에 접근
+ `__len__()`: 그룹 안의 데이터 수 반환
+ `__getitem__()`: Index를 이미지의 이름으로 변환하고 이미지 데이터 반환

## Discriminator

~~~python
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            View(218*178*3),
            nn.Linear(3*218*178, 100),
            nn.LeakyReLU(),
            nn.LayerNorm(100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.0001)
        self.optimiser.param_groups[0]['capturable'] = True
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
        if (self.counter % 1000 == 0):
            print("counter = ", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns = ['loss'])
        df.plot(ylim = (0), figsize = (16, 8), alpha = 0.1, marker = '.', grid = True, yticks = (0, 0.2, 0.5, 1))
        pass
    pass
~~~

+ 이미지가 실제인지 생성된 이미지인지 분류
+ 입력: $218\times178\times3=116412$
  + 신경망이 완전 연결 신경망이므로 일관된 기준을 통해 정렬
  + 이미지를 어떻게 풀어서 정렬하는지는 중요하지 않음
+ `View()`: 3차원 이미지 텐서를 1차원 형태의 텐서로 변환
  + $(218,178,3)\rightarrow(218\times178\times3)$
  + `nn.Module`을 상속하여 `Sequential` 내에서 다른 모듈과 함께 사용 가능

~~~python
%%time

D = Discriminator()
D.to(device)

for image_data_tensor in celeba_dataset:
    D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
    D.train(generate_random_image((218,178,3)), torch.cuda.FloatTensor([0.0]))
    pass
~~~

> Discriminator test
![discriminator-test](/images/generative-adversarial-network-3/discriminator-test.png)

## Generator

~~~python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 3*10*10),
            nn.LeakyReLU(),
            nn.LayerNorm(3*10*10),
            nn.Linear(3*10*10, 3*218*178),
            nn.Sigmoid(),
            View((218, 178, 3))
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.0001)
        self.optimiser.param_groups[0]['capturable'] = True
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
        df.plot(ylim = (0), figsize = (16, 8), alpha = 0.1, marker = '.', grid = True, yticks = (0, 0.2, 0.5, 1))
        pass
    pass
~~~

+ 3차원의 텐서를 $(218,178,3)$의 크기로 결과를 출력하도록 수정
+ $100\rightarrow300\rightarrow3\times218\times178\rightarrow(218,178,3)$

> Result of generator
![result-of-generator](/images/generative-adversarial-network-3/result-of-generator.png)

## GAN Training

~~~python
%%time

D = Discriminator()
D.to(device)
G = Generator()
G.to(device)

epochs = 1

for epoch in range(epochs):
    for image_data_tensor in celeba_dataset:
        D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
        D.train(G.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))
        G.train(D, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))
        pass
    pass
~~~

> Loss of GAN training process
![loss-of-gan-training-process-2](/images/generative-adversarial-network-3/loss-of-gan-training-process-2.png)

> Results of generator
![results-of-generator-2](/images/generative-adversarial-network-3/results-of-generator-2.png)

+ 생성기는 직접 이미지를 통해 훈련하지 않고 이미지를 생성할 때 훈련 데이터의 우도 (likelihood) 사용