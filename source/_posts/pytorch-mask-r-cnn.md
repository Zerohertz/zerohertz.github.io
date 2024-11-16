---
title: Implementing Mask R-CNN with PyTorch
date: 2023-04-05 21:34:10
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# Introduction

<details>
<summary>
<a href="https://github.com/Zerohertz/Mask_R-CNN">Mask R-CNN</a>?
</summary>

Mask R-CNN은 Faster R-CNN에 Segmentation 네트워크를 추가한 딥러닝 알고리즘으로, 객체 검출 (Object detection)과 분할을 모두 수행할 수 있습니다.
기존 Faster R-CNN은 RPN (Region Proposal Network)을 사용하여 객체의 경계 상자 (Bounding box)를 추출하고, 추출된 경계 상자를 입력으로 사용하여 객체 인식을 수행합니다. 이러한 방식은 객체의 위치와 클래스 정보를 검출할 수 있지만, 객체 내부의 픽셀-레벨 Segmentation 정보는 제공하지 않습니다.
Mask R-CNN은 Faster R-CNN의 RPN 뿐만 아니라, RoIAlign (Rectangle of Interest Alignment)을 사용하여 추출된 경계 상자 내부의 픽셀-레벨 Segmentation 정보를 추출할 수 있는 분할 네트워크를 추가합니다. 이를 통해, 객체 검출과 동시에 객체 내부의 픽셀-레벨 Segmentation 정보를 추출할 수 있습니다.
또한, Mask R-CNN은 이를 위해 Faster R-CNN과 함께 사용되는 합성곱 신경망 (Convolutional Neural Network)을 미세 조정 (Fine-tuning)하여 분할 네트워크의 성능을 최적화합니다.
Mask R-CNN은 객체 검출과 분할 작업에서 매우 강력한 성능을 보여주며, COCO (Common Objects in Context) 데이터셋에서 현재 가장 높은 정확도를 보이고 있습니다. 따라서, 객체 검출과 분할이 모두 필요한 다양한 응용 분야에서 활용되고 있습니다.

</details>

```bash
├── makeGT.py
├── model
│   ├── __init__.py
│   ├── load_data.py
│   ├── model.py
│   ├── README.md
│   ├── test.py
│   └── train.py
├── README.md
├── requirements.txt
├── test.py
├── train.py
└── utils
    ├── coco_eval.py
    ├── coco_utils.py
    ├── engine.py
    ├── __init__.py
    ├── README.md
    ├── transforms.py
    └── utils.py
```

Mask R-CNN의 training, test, visualization, evaluation을 진행할 수 있게 PyTorch를 사용하여 위와 같은 구조로 개발하는 과정을 기록한다.

사용될 데이터는 [ISIC 2016 Challenge - Task 3B: Segmented Lesion Classification](https://challenge.isic-archive.com/landing/2016/41/)이며 예시는 아래와 같다.

```bash
├── ISBI2016_ISIC_Part3B_Test_Data
│   ├── ISIC_0000003.jpg
│   ├── ISIC_0000003_Segmentation.png
│   └── ...
├── ISBI2016_ISIC_Part3B_Training_Data
│   ├── ISIC_0000000.jpg
│   ├── ISIC_0000000_Segmentation.png
│   └── ...
├── ISBI2016_ISIC_Part3B_Test_GroundTruth.csv
└── ISBI2016_ISIC_Part3B_Training_GroundTruth.csv
```

<!-- More -->

![image](/images/pytorch-mask-r-cnn/229952708-4db35f2b-ce8b-4a4f-b7c4-f84a9dcaf2ae.png)

이 데이터는 두 가지 클래스 (`benign`, `malignant`)로 구성되어 있고 위 사진에서 알 수 있는 것처럼 분할된 mask를 함께 제공한다.
Mask R-CNN이 Segmentation 정보를 학습 및 테스트할 수 있도록 `TrainingData`와 `TestData`를 구성했고, 그를 위한 코드는 아래와 같다.

```python saveData.py
import os
import shutil

import cv2
import pandas as pd


def initializeData(DataStoreName):
    tmp = os.getcwd()
    if DataStoreName in os.listdir():
        shutil.rmtree(DataStoreName)
    os.mkdir(DataStoreName)
    os.chdir(DataStoreName)
    os.mkdir('images')
    os.mkdir('masks')
    os.chdir(tmp)
    return (tmp + '/' + DataStoreName + '/' + 'images/', tmp + '/' + DataStoreName + '/' + 'masks/')

def saveData(target, ImgDir, MaskDir, label):
    # Make Target Data: IMG
    shutil.copy(target, ImgDir + target)
    # Make Target Data: Mask (GT)
    mask = cv2.imread(target.replace('.jpg', '_Segmentation.png'), cv2.IMREAD_UNCHANGED)
    mask[mask == 255] = label
    cv2.imwrite(MaskDir + target.replace('jpg', 'png'), mask)

if __name__ == "__main__":
    ImgDir, MaskDir = initializeData('TrainingData')
    target = 'ISBI2016_ISIC_Part3B_Training_Data'
    GT = pd.read_csv(target.replace('Data', 'GroundTruth.csv'), header=None, index_col=0)
    enc = {}
    for i, j in enumerate(GT[1].unique()):
        enc[j] = i + 1
    print('='*10, enc, '='*10)

    os.chdir(target)
    for tmp in os.listdir():
        if (not ('_Segmentation' in tmp)) and ('.jpg' in tmp):
            saveData(tmp, ImgDir, MaskDir, enc[GT.loc[tmp[:-4], 1]])

    os.chdir('..')
    ImgDir, MaskDir = initializeData('TestData')
    target = 'ISBI2016_ISIC_Part3B_Test_Data'
    GT = pd.read_csv(target.replace('Data', 'GroundTruth.csv'), header=None, index_col=0)
    enc = {}
    for i, j in enumerate(GT[1].unique()):
        enc[j] = i + 1
    print('='*10, enc, '='*10)

    os.chdir(target)
    for tmp in os.listdir():
        if (not ('_Segmentation' in tmp)) and ('.jpg' in tmp):
            saveData(tmp, ImgDir, MaskDir, enc[GT.loc[tmp[:-4], 1]])
```

위의 코드를 실행하면 아래와 같이 학습 및 테스트를 위한 데이터 디렉토리를 구성할 수 있다.

```bash
├── TestData
│   ├── images
│   │   ├── ISIC_0000013.jpg
│   │   ├── ISIC_0000015.jpg
│   │   └── ...
│   └── masks
│       ├── ISIC_0000013.png
│       ├── ISIC_0000015.png
│       └── ...
└── TrainingData
    ├── images
    │   ├── ISIC_0000001.jpg
    │   ├── ISIC_0000002.jpg
    │   └── ...
    └── masks
        ├── ISIC_0000001.png
        ├── ISIC_0000002.png
        └── ...
```

# Customized Dataset

이렇게 구성된 Dataset을 Load하기 위해 `CustomizedDatset`이라는 클래스를 개발해야한다.

```python model/load_data.py
import os
from PIL import Image

import numpy as np
from torchvision.transforms import Normalize

import torch


class CustomizedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.Normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

PyTorch로 구성된 Mask R-CNN이 학습 및 테스트 시 사용할 수 있기 위해 `torch.utils.data.Dataset`을 상속하였다.
인스턴스를 생성할 때 `root`를 입력받고, `images` 디렉토리와 `masks` 디렉토리 내부의 `os.listdir()`로 이미지와 마스크의 리스트들을 프로퍼티로 입력한다.
또한 학습 시 데이터 증강을 위해 `transforms` 프로퍼티를, 정규화를 위한 `Normalize` 프로퍼티를 추가했다.

```python model/load_data.py
class CustomizedDataset(torch.utils.data.Dataset):
    ...
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            img = self.Normalize(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
```

가장 중요한 부분인 `__getitem__()` 메서드에서는 인덱스를 입력해주면 이미지 (`img`)와 메타 정보들 (`target`)을 출력해준다.
인스턴스 생성 시 정렬된 이미지와 마스크들 리스트 (`self.imgs`, `self.masks`)와 `self.root`를 통해 절대 경로 (`img_path`, `mask_path`)를 산출하고 `PIL.Image`로 각 해당하는 이미지를 불러왔다.
이후 Numpy로 간단한 데이터 핸들링을 거치고 `torch.tensor`로 변환 후 각 메타 정보에 해당하는 값들을 딕셔너리인 `target`에 입력해준 뒤 리턴해준다.
이렇게 개발된 `CustomizedDataset`의 예시는 아래와 같다.

```python
>>> from model import CustomizedDataset
>>> c = CustomizedDataset("../data/TrainingData")
>>> c[0]
(<PIL.Image.Image image mode=RGB size=1022x767 at 0x7F7E7DBDB5E0>, {'boxes': tensor([[ 51.,  47., 898., 634.]]), 'labels': tensor([1]), 'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8), 'image_id': tensor([0]), 'area': tensor([497189.]), 'iscrowd': tensor([0])})
>>> c[3]
(<PIL.Image.Image image mode=RGB size=1022x767 at 0x7F7E7DBDBEB0>, {'boxes': tensor([[181.,  57., 718., 717.]]), 'labels': tensor([2]), 'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8), 'image_id': tensor([3]), 'area': tensor([354420.]), 'iscrowd': tensor([0])})
```

`CustomizedDataset`의 데이터 증강을 위한 `get_transform()` 함수는 아래와 같이 구성되어있다.

```python model/load_data.py
from utils import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
```

마지막으로 학습 과정에서 학습과 테스트를 위한 `torch.utils.data.DataLoader`를 한번에 불러올 수 있는 `load_data()` 함수를 개발했다.

```python model/load_data.py
from utils import utils


def load_data(TrainingDir, TestDir, batch_size=8, num_workers=16):
    TrainingDataset = CustomizedDataset(TrainingDir, get_transform(train=True))
    TestDataset = CustomizedDataset(TestDir, get_transform(train=False))

    TrainingDataset = torch.utils.data.DataLoader(
        TrainingDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    TestDataset = torch.utils.data.DataLoader(
        TestDataset, batch_size=batch_size//2, shuffle=False, num_workers=num_workers//2,
        collate_fn=utils.collate_fn)

    return TrainingDataset, TestDataset
```

---

# Init Model

학습 및 테스트를 위한 데이터들은 준비를 완료했으니 학습 및 테스트를 할 모델을 구축해야한다.
따라서 아래와 같이 `init_model()` 함수를 통해 모델을 생성해줄 수 있다.

```python model/model.py
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def init_model(device, num_classes):
    '''
    Mask R-CNN
    '''
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 2048
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
    return model.to(device)
```

`maskrcnn_resnet50_fpn(weights="DEFAULT")` 메서드를 통해 COCO로 pre-trained 모델을 불러오고 분류기를 재정의해준다.
이는 bbox 및 mask 예측 모듈의 클래스 수가 변하기 때문에 신경망 구조가 필연적으로 변경되기 때문이다.
또한 모델의 원활한 학습과 테스트를 위해 [utils](https://github.com/Zerohertz/Mask_R-CNN/tree/master/utils) 모듈을 준비한다.

---

# Train

학습 및 테스트를 위한 데이터를 준비했고, 모델 또한 준비를 했으니 이제는 학습을 할 수 있다.
학습을 아래와 같이 구성할 수 있다.

```python train.py
import argparse

import torch

from model import init_model, prepare_training, load_data, train


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    return parser.parse_args()

def main():
    args = opts()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### Prepare Dataset #####
    TrainingDataset, TestDataset = load_data("../data/TrainingData",
                                             "../data/TestData",
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    ###### Prepare Training #####
    config = prepare_training(model)
    config.update({'device': device,
                'TrainingDataset': TrainingDataset,
                'TestDataset': TestDataset,
                'num_epochs': args.epoch})

    ##### Training #####
    train(model, **config)

if __name__ == "__main__":
    main()
```

우선적으로 학습 코드를 실행시킬 때 몇가지 옵션을 간단히 수정하기 위해 `argparse` 모듈 기반 `opts()` 함수를 개발하였다.
이렇게 정의된 변수를 통해 학습 및 테스트 데이터, 모델, 그리고 학습을 위한 조건을 정의하였고 최종적으로 `model.train.train()` 함수로 학습을 진행한다.
해당 함수는 아래와 같이 구성된다.

```python model/train.py
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils.engine import train_one_epoch, evaluate
from .test import get_results


def train(model,
        device,
        optimizer,
        lr_scheduler,
        TrainingDataset,
        TestDataset,
        num_epochs=2):
    writer = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        lr, loss_dict, loss = train_one_epoch(model, optimizer, TrainingDataset, device, epoch, print_freq=10)
        writer.add_scalar('lr', lr, epoch)
        for k in loss_dict:
            writer.add_scalar(k, loss_dict[k], epoch)
        writer.add_scalar('loss', loss, epoch)
        lr_scheduler.step()
        CocoEvaluator = evaluate(model, TestDataset, device=device)
        res = get_results(CocoEvaluator)
        for i, j in res:
            writer.add_scalar(i, j, epoch)
        if epoch % 20 == 9:
            torch.save(model.state_dict(), './' + str(epoch + 1) + 'ep.pth')
    torch.save(model.state_dict(), './' + str(epoch + 1) + 'ep.pth')
```

학습 과정을 모니터링하기 위하여 [TensorBoard](https://github.com/Zerohertz/Mask_R-CNN/issues/12)를 기용하여 매 epoch 마다의 learning rate, 모델 내 다양한 loss, 최종 loss, precision, recall을 출력할 수 있도록 하였다.
TensorBoard의 시각화 예시는 아래와 같다.

![TensorBoard](/images/pytorch-mask-r-cnn/230006393-06921cf0-2e20-45b5-9984-a2928a9166a2.png)

+ <span style='color: #0000FF'>SGD (lr=0.001, step_size=20)</span>
+ <span style='color: #000000'>Adam (lr=0.001, step_size=30)</span>
+ <span style='color: #FF0000'>Adam (lr=0.001, step_size=20)</span>
+ <span style='color: #800a0a'>Adam (lr=0.001, step_size=20, Normalize)</span>

또한 학습 과정 중 가중치를 저장하기 위해 매 20 epoch 마다 `torch.save()`를 실행할 수 있도록 개발했다.

---

# Test

이제 온전히 학습된 가중치가 있으니 테스트 데이터에 대해 테스트를 진행할 수 있다.
테스트는 아래와 같이 진행할 수 있다.

```python test.py
import argparse

import torch

from model import init_model, test


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)
    parser.add_argument("--exp", default="test", type=str)
    return parser.parse_args()

def main():
    args = opts()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##### INIT Mask R-CNN #####
    num_classes = 3
    model = init_model(device, num_classes)

    model.load_state_dict(torch.load(args.weights))
    model.cuda()
    model.eval()

    ##### Test #####
    test("../data/TestData",
         args.exp,
         model,
         device,
         {1: 'benign', 2: 'malignant'})

if __name__ == "__main__":
    main()
```

학습 코드와 유사하게 `opts()` 함수로 테스트 시 변수들을 입력할 수 있게 개발했고, 테스트 데이터 및 모델 불러온 후 최종적으로 `model.test.test()` 함수로 테스트를 진행한다.
테스트 코드를 개발하기 위해서는 두 가지가 필요하다.

1. 모델의 출력을 시각화할 수 있는 함수
2. 모델의 출력을 정량적으로 평가할 수 있는 함수

따라서 `model.test.test()` 함수는 아래와 같이 개발하였다.

```python model/test.py
import os
import shutil
import random
import csv

import numpy as np

import torch
from torchvision.ops import nms
import cv2

from tqdm import tqdm

from utils import utils
from utils.engine import evaluate
from .load_data import get_transform, CustomizedDataset


def draw_gt(img_path, obj):
    if not 'exp' in os.listdir():
        os.mkdir('exp')
    if not 'Ground_Truth' in os.listdir('exp'):
        os.mkdir('exp/Ground_Truth')
    else:
        shutil.rmtree('exp/Ground_Truth')
        os.mkdir('exp/Ground_Truth')
    TestDataset = CustomizedDataset(img_path, get_transform(train=False))
    for i, tmp in enumerate(tqdm(TestDataset)):
        img = cv2.imread(img_path + '/images/' + TestDataset.imgs[i])
        img = np.array(img)
        boxes, masks, labels = tmp[1]['boxes'], tmp[1]['masks'], tmp[1]['labels']
        for box, mask, label in zip(boxes, masks, labels):
            box, mask, label = box.numpy(), mask.numpy(), int(label)
            try:
                label = obj[label]
            except:
                label = 'Unknown: ' + str(label)
            color = (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
            thickness = 2
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label_text = f"{label}"
            cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,0), 5)
            cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, color, thickness)
            mask = (mask > 0.5)
            masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            no_masked_img = cv2.bitwise_and(img, img, mask=255-mask.astype(np.uint8))
            masked_img[np.where((masked_img != [0, 0, 0]).all(axis=2))] = color
            img = cv2.addWeighted(masked_img, 0.5, no_masked_img, 1, 0)
        cv2.imwrite('exp/Ground_Truth/' + TestDataset.imgs[i], img)

def get_results(CocoEvaluator):
    '''
    CocoEvaluator: utils.coco_eval.CocoEvaluator
    '''
    keys = [
        "Precision: IoU=0.50:0.95/area=all/maxDets=100",
        "Precision: IoU=0.50/area=all/maxDets=100",
        "Precision: IoU=0.75/area=all/maxDets=100",
        "Precision: IoU=0.50:0.95/area=small/maxDets=100",
        "Precision: IoU=0.50:0.95/area=medium/maxDets=100",
        "Precision: IoU=0.50:0.95/area=large/maxDets=100",
        "Recall: IoU=0.50:0.95/area=all/maxDets=1",
        "Recall: IoU=0.50:0.95/area=all/maxDets=10",
        "Recall: IoU=0.50:0.95/area=all/maxDets=100",
        "Recall: IoU=0.50:0.95/area=small/maxDets=100",
        "Recall: IoU=0.50:0.95/area=medium/maxDets=100",
        "Recall: IoU=0.50:0.95/area=large/maxDets=100"
    ]
    res = []
    bbox_res = CocoEvaluator.coco_eval['bbox'].stats
    segm_res = CocoEvaluator.coco_eval['segm'].stats
    for i, j in zip(keys, bbox_res):
        res.append(("Bbox - " + i, j))
    for i, j in zip(keys, segm_res):
        res.append(("Segm - " + i, j))
    return res

def init_output(output):
    idx = nms(output[0]['boxes'], output[0]['scores'], 0.2)
    boxes = output[0]['boxes'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    labels = output[0]['labels'].cpu().detach().numpy()
    masks = output[0]['masks'].cpu().detach().numpy()
    return idx, boxes, scores, labels, masks

def draw_res(img_path, img_f, tar_path, output, obj={}):
    '''
    img_path: Path of Target Image
    output: Output of Mask R-CNN (Input: Target Image)
    obj: Actual Label According to Model Label in Dictionary
    '''
    idx, boxes, scores, labels, masks = init_output(output)
    img = cv2.imread(img_path + img_f)
    img = np.array(img)
    for i in idx:
        box, mask, score, label = boxes[i], masks[i], scores[i], labels[i]
        if score < 0.5:
            continue
        try:
            label = obj[label]
        except:
            label = 'Unknown: ' + str(label)
        color = (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256))
        thickness = 2
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,0), 5)
        cv2.putText(img, label_text, (x1, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 2, color, thickness)
        mask = (mask > 0.5)
        masked_img = cv2.bitwise_and(img, img, mask=mask[0].astype(np.uint8))
        no_masked_img = cv2.bitwise_and(img, img, mask=255-mask[0].astype(np.uint8))
        masked_img[np.where((masked_img != [0, 0, 0]).all(axis=2))] = color
        img = cv2.addWeighted(masked_img, 0.5, no_masked_img, 1, 0)
    cv2.imwrite('exp/' + tar_path + '/' + img_f, img)

def test(TestDataset_path, tar_path, model, device, obj={}):
    if not 'exp' in os.listdir():
        os.mkdir('exp')
    if not tar_path in os.listdir('exp'):
        os.mkdir('exp/' + tar_path)
    else:
        shutil.rmtree('exp/' + tar_path)
        os.mkdir('exp/' + tar_path)
    TestDataset = CustomizedDataset(TestDataset_path, get_transform(train=False))
    with torch.no_grad():
        for i, tmp in enumerate(tqdm(TestDataset)):
            output = model(tmp[0].unsqueeze_(0).to(device))
            draw_res(TestDataset_path + '/images/',
                     TestDataset.imgs[i],
                     tar_path,
                     output,
                     obj)
    TestDataset = torch.utils.data.DataLoader(
        TestDataset, batch_size=8, shuffle=False, num_workers=16,
        collate_fn=utils.collate_fn)
    CocoEvaluator = evaluate(model, TestDataset, device=device)
    res = get_results(CocoEvaluator)
    with open('./exp/' + tar_path + '/res.csv', 'a', encoding='utf-8') as f:
        wr = csv.writer(f)
        for i, j in res:
            wr.writerow([i, j])
```

먼저 `draw_gt()` 함수는 기본적으로 모델 테스트 결과와 비교하기 위해 개발했다.
데이터 내의 mask와 label을 실제 사진에 입히고 시각화하는 함수다.
다음으로 `get_results()` 함수는 `CocoEvaluator`로 평가된 값들을 불러오고 TensorBoard에 출력할 수 있도록 데이터를 핸들링해주는 함수다.
입력으로 평가가 완료된 `utils.coco_eval.CocoEvaluator` 인스턴스를 받으면 내부의 결과 값들을 불러오고 해당하는 평가 지표의 이름과 함께 리턴해준다.
`init_output()` 함수는 Mask R-CNN의 결과를 [NMS](https://github.com/Zerohertz/Mask_R-CNN/issues/15)로 후처리해주고, 출력된 결과를 CPU로 이동 후 Numpy 배열로 변환하고 리턴해준다.
`draw_res()` 함수는 `init_output()` 함수에서 정리된 결과를 토대로 시각화하는 함수다.
`draw_gt()` 함수와 같은 양식으로 시각화할 수 있도록 개발했다.
최종적으로 `test()` 함수에서 테스트 데이터에 대해 결과를 산출하고, 시각화한 뒤 정량적으로 평가하여 `.csv` 형식으로 저장하는 것을 확인할 수 있다.
`draw_gt()` 함수와 `draw_res()` 함수를 통해 출력한 결과의 예시는 아래와 같다.

![draw()](/images/pytorch-mask-r-cnn/230010864-66dcc3dc-581d-4dec-91a7-dd536c4ab17a.png)

---

# Reference

+ [PyTorch](https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html)