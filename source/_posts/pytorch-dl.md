---
title: Deep Learning with PyTorch
date: 2023-01-17 18:07:18
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# Load Data

~~~python In[1]
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/car_evaluation.csv')
dataset.head()
~~~

<img src="/images/pytorch-dl/dataset.head.png" alt="dataset.head" width="832" />

~~~python In[2]
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind = 'pie', autopct = '%0.05f%%', colors = ['lightblue', 'lightgreen', 'orange', 'pink'], explode = (0.05, 0.05, 0.05, 0.05))
~~~

![results](/images/pytorch-dl/results.png)

<!-- More -->

---

# Data Tensorization

~~~python In[3]
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')
    
price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)

categorical_data = torch.tensor(categorical_data, dtype = torch.int64)
categorical_data[:10]
~~~

~~~python Out[3]
tensor([[3, 3, 0, 0, 2, 1],
        [3, 3, 0, 0, 2, 2],
        [3, 3, 0, 0, 2, 0],
        [3, 3, 0, 0, 1, 1],
        [3, 3, 0, 0, 1, 2],
        [3, 3, 0, 0, 1, 0],
        [3, 3, 0, 0, 0, 1],
        [3, 3, 0, 0, 0, 2],
        [3, 3, 0, 0, 0, 0],
        [3, 3, 0, 1, 2, 1]])
~~~

~~~python In[4]
outputs = pd.get_dummies(dataset.output)

output_acc = outputs.acc.values
output_good = outputs.good.values
output_unacc = outputs.unacc.values
output_vgood = outputs.vgood.values

output_acc = torch.tensor(output_acc, dtype = torch.int64)
output_good = torch.tensor(output_good, dtype = torch.int64)
output_unacc = torch.tensor(output_unacc, dtype = torch.int64)
output_vgood = torch.tensor(output_vgood, dtype = torch.int64)

print(categorical_data.shape)
print(output_acc.shape)
print(output_acc[:20])
~~~

~~~python Out[4]
torch.Size([1728, 6])
torch.Size([1728])
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
~~~

---

# Embedding Setup

~~~python In[5]
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in categorical_column_sizes]

print(categorical_embedding_sizes)
~~~

~~~python Out[5]
[(4, 2), (4, 2), (4, 2), (3, 2), (3, 2), (3, 2)]
~~~

~~~python In[6]
print(dataset[categorical_columns[0]])
print('-' * 60)
print(dataset[categorical_columns[0]].cat)
print('-' * 60)
print(dataset[categorical_columns[0]].cat.categories)
~~~

~~~python Out[6]
0       vhigh
1       vhigh
2       vhigh
3       vhigh
4       vhigh
        ...  
1723      low
1724      low
1725      low
1726      low
1727      low
Name: price, Length: 1728, dtype: category
Categories (4, object): ['high', 'low', 'med', 'vhigh']
------------------------------------------------------------
<pandas.core.arrays.categorical.CategoricalAccessor object at 0x1521522b0>
------------------------------------------------------------
Index(['high', 'low', 'med', 'vhigh'], dtype='object')
~~~

---

# Data Segmentation

~~~python In[7]
total_records = len(dataset)
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records:]
train_output_acc = output_acc[:total_records - test_records]
test_output_acc = output_acc[total_records - test_records:]
train_output_good = output_good[:total_records - test_records]
test_output_good = output_good[total_records - test_records:]
train_output_unacc = output_unacc[:total_records - test_records]
test_output_unacc = output_unacc[total_records - test_records:]
train_output_vgood = output_vgood[:total_records - test_records]
test_output_vgood = output_vgood[total_records - test_records:]

print(len(categorical_train_data))
print(len(train_output_acc))
print(len(categorical_test_data))
print(len(test_output_acc))
~~~

~~~python Out[7]
1383
1383
345
345
~~~

---

# Model Construction

~~~python In[8]
class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p = 0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        
        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols
        
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace = True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i
        
        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)
    
    def forward(self, x_categorical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x

model = Model(categorical_embedding_sizes, 4, [200, 100, 50], p = 0.4)
print(model)
~~~

~~~python Out[8]
Model(
  (all_embeddings): ModuleList(
    (0): Embedding(4, 2)
    (1): Embedding(4, 2)
    (2): Embedding(4, 2)
    (3): Embedding(3, 2)
    (4): Embedding(3, 2)
    (5): Embedding(3, 2)
  )
  (embedding_dropout): Dropout(p=0.4, inplace=False)
  (layers): Sequential(
    (0): Linear(in_features=12, out_features=200, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.4, inplace=False)
    (4): Linear(in_features=200, out_features=100, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.4, inplace=False)
    (8): Linear(in_features=100, out_features=50, bias=True)
    (9): ReLU(inplace=True)
    (10): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout(p=0.4, inplace=False)
    (12): Linear(in_features=50, out_features=4, bias=True)
  )
)
~~~

---

# Training Options Setup

~~~python In[9]
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

if torch.cuda.is_available(): # CUDA
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device
~~~

~~~python Out[9]
device(type='mps')
~~~

---

# Training!

~~~python In[10]
%%time

epochs = 1000
aggregated_losses = []
train_output_acc = train_output_acc.to(device = device, dtype = torch.int64)

for i in range(epochs):
    y_pred = model(categorical_train_data).to(device)
    single_loss = loss_function(y_pred, train_output_acc)
    aggregated_losses.append(single_loss)
    
    if i % 25 == 0:
        print(f'epoch: {i + 1:3} loss: {single_loss.item():10.8f}')
    
    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()
print(f'epoch: {i + 1:3} loss: {single_loss.item():10.10f}')
~~~

~~~python Out[10]
epoch:   1 loss: 1.48282647
epoch:  26 loss: 1.19633317
epoch:  51 loss: 1.05753756
epoch:  76 loss: 0.94815171
epoch: 101 loss: 0.83707392
epoch: 126 loss: 0.71467799
epoch: 151 loss: 0.62778354
epoch: 176 loss: 0.51973403
epoch: 201 loss: 0.47347799
...
epoch: 901 loss: 0.26305360
epoch: 926 loss: 0.25292999
epoch: 951 loss: 0.25285697
epoch: 976 loss: 0.25961897
epoch: 1000 loss: 0.2632845342
CPU times: user 6.97 s, sys: 4.82 s, total: 11.8 s
Wall time: 8.21 s
~~~

---

# Validation

~~~python In[11]
test_output_acc = test_output_acc.to(device = device, dtype = torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val.to(device), test_output_acc)

print(f'Loss: {loss:.8f}')
~~~

~~~python Out[11]
Loss: 0.49808758
~~~

~~~python In[12]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

test_output_acc = test_output_acc.to('cpu')
print(confusion_matrix(test_output_acc, y_val))
print(classification_report(test_output_acc, y_val))
print(accuracy_score(test_output_acc, y_val))
~~~

~~~python Out[12]
[[230  52]
 [ 40  23]]
              precision    recall  f1-score   support

           0       0.85      0.82      0.83       282
           1       0.31      0.37      0.33        63

    accuracy                           0.73       345
   macro avg       0.58      0.59      0.58       345
weighted avg       0.75      0.73      0.74       345

0.7333333333333333
~~~