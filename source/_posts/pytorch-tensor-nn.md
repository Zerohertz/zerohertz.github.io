---
title: Tensor & Neural Network in PyTorch
date: 2021-09-09 17:31:51
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
- scikit-learn
---
# Import Packages

~~~python
>>> import torch
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from torch.autograd import Variable
~~~

# Tensor

## Scala (0-Dim Tensor)

~~~python
>>> x = torch.rand(10)
>>> x.size()
torch.Size([10])
>>> x
tensor([0.8956, 0.5085, 0.8703, 0.2497, 0.2278, 0.9197, 0.5894, 0.1974, 0.6448,
        0.6871])
~~~

<!-- More -->

## Vector (1-Dim Tensor)

~~~python
>>> temp = torch.FloatTensor([2,3,1,5.4,2,5.8])
>>> temp.size()
torch.Size([6])
>>> temp
tensor([2.0000, 3.0000, 1.0000, 5.4000, 2.0000, 5.8000])
~~~

## Matrix (2-Dim Tensor)

~~~python
>>> from sklearn.datasets import load_boston
>>> boston = load_boston()
>>> print(boston.data.shape)
(506, 13)
>>> boston.feature_names
array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
>>> boston_tensor = torch.from_numpy(boston.data)
>>> boston_tensor.size()
torch.Size([506, 13])
>>> boston_tensor[:2]
tensor([[6.3200e-03, 1.8000e+01, 2.3100e+00, 0.0000e+00, 5.3800e-01, 6.5750e+00,
         6.5200e+01, 4.0900e+00, 1.0000e+00, 2.9600e+02, 1.5300e+01, 3.9690e+02,
         4.9800e+00],
        [2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01, 6.4210e+00,
         7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02, 1.7800e+01, 3.9690e+02,
         9.1400e+00]], dtype=torch.float64)
~~~

## 3-Dim Tensor

~~~python
>>> from PIL import Image
>>> cat = np.array(Image.open('cat.jpg').resize((224,224)))
>>> cat_tensor = torch.from_numpy(cat)
>>> cat_tensor.size()
torch.Size([224, 224, 3])
>>> plt.imshow(cat)
<matplotlib.image.AxesImage object at 0x7fa7481d4580>
>>> plt.show()
~~~

<img src="/images/pytorch-tensor-nn/cat.png" alt="cat" width="752" />

## Tensor Slicing

~~~python
>>> sales = torch.FloatTensor([1000.0,323.2,333.4,444.5,1000.0,323.2,333.4,444.5])
>>> sales[:5]
tensor([1000.0000,  323.2000,  333.4000,  444.5000, 1000.0000])
>>> sales[:-5]
tensor([1000.0000,  323.2000,  333.4000])
>>> plt.imshow(cat_tensor[:,:,0].numpy())
<matplotlib.image.AxesImage object at 0x7fa74962bb20>
>>> plt.show()
~~~

<img src="/images/pytorch-tensor-nn/cat-tensor.png" alt="cat-tensor" width="752" />

~~~python
>>> plt.imshow(cat_tensor[25:175,60:130,0].numpy())
<matplotlib.image.AxesImage object at 0x7fc9826d85e0>
>>> plt.show()
~~~

<img src="/images/pytorch-tensor-nn/cat-tensor-slicing.png" alt="cat-tensor-slicing" width="752" />

***

# Neural Network

## Variable

~~~python
>>> from torch.autograd import Varialbe
>>> x = Variable(torch.ones(2,2),requires_grad=True)
>>> y = x.mean()
>>> y.backward()
>>> x.grad
tensor([[0.2500, 0.2500],
        [0.2500, 0.2500]])
>>> x.grad_fn
>>> x.data
tensor([[1., 1.],
        [1., 1.]])
>>> y.grad_fn
<MeanBackward0 object at 0x7fc9826cb550>
~~~

## Building Network

~~~python
def get_data():
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.564,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,3.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype),requires_grad=False).view(17,1)
    y = Variable(torch.from_numpy(train_Y).type(dtype),requires_grad=False)
    return X,y

def get_weights():
    w = Variable(torch.randn(1),requires_grad=True)
    b = Variable(torch.randn(1),requires_grad=True)
    return w,b
~~~

## Network Implementation

~~~python
>>> def network(x):
...     y_pred = torch.matmul(x,w)+b
...     return y_pred
...
>>> import torch.nn as nn
>>> f = nn.Linear(17,1)
>>> f
Linear(in_features=17, out_features=1, bias=True)
~~~

## Loss Function

~~~python
def loss_fn(y,y_pred):
    loss = (y_pred-y).pow(2).sum()
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()
    return loss.data
~~~

## Optimizer

~~~python
def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data
~~~

## Dataset implementation

~~~python
from torch.utils.data import Dataset

class DogsAndCatsDataset(Dataset):
    def __init__(self):
        self.files = glob(root_dir)
        self.size = size
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]
        return img,label
~~~

## Dataloader

~~~python
from torch.utils.data import Dataset, DataLoader

dataloader = DataLoader(DagsAndCatDataset,batch_size=32,num_workers=2)
for imgs,labels in dataloader:
        # Training Network
~~~

## Simple Neural Network

~~~python
import torch
from torch.autograd import Variable
import numpy as np


def get_data():
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.564,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,3.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype),requires_grad=False).view(17,1)
    y = Variable(torch.from_numpy(train_Y).type(dtype),requires_grad=False)
    return X,y

def get_weights():
    w = Variable(torch.randn(1),requires_grad=True)
    b = Variable(torch.randn(1),requires_grad=True)
    return w,b

def network(x):
    y_pred = torch.matmul(x,w)+b
    return y_pred

def loss_fn(y,y_pred):
    loss = (y_pred-y).pow(2).sum()
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()
    return loss.data

def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


learning_rate = 1e-4
x,y = get_data()
w,b = get_weights()
for i in range(500):
    y_pred = network(x)
    loss = loss_fn(y,y_pred)
    if i % 50 == 0:
        print(loss)
    optimize(learning_rate)
~~~

> Result

~~~python
tensor(42.0655)
tensor(11.9596)
tensor(11.6388)
tensor(11.3310)
tensor(11.0358)
tensor(10.7524)
tensor(10.4806)
tensor(10.2198)
tensor(9.9695)
tensor(9.7294)

Process finished with exit code 0
~~~