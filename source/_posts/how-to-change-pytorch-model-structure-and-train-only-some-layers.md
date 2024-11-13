---
title: How to Change PyTorch Model Structure and Train Only Some Layers
date: 2023-03-09 10:15:22
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# Introduction

논문의 저자가 제공하거나 논문을 참고하여 개발된 모델은 보통 config 파일 (e.g. `config.yaml`, `config.py`)이 존재하고, 해당 파일을 통해 [이렇게](https://github.com/whai362/pan_pp.pytorch/blob/master/config/pan_pp/pan_pp_r18_ic15_736_finetune.py#L8) 모델 구조를 변경할 수 있다.
하지만 기존의 소스에 본인이 원하는 모델 구조가 없다면 어떻게 개발하는지, 그리고 기존에 없던 레이어를 어떻게 훈련하면 좋을지 알아보자.
이 글에서는 [이 논문](https://arxiv.org/abs/2105.00405)을 기반으로 개발된 모델인 [whai362/pan_pp.pytorch](https://github.com/whai362/pan_pp.pytorch)를 기준으로 개발하겠다.
간단한 목표 설정을 해보기 위해 대략적인 모델의 설명을 진행하겠다.

## PAN++

PAN++는 STR (Scene Text Recognition)을 위해 개발되었지만, 본 글에서는 STD (Scene Text Detection) 부분까지만 사용하며 해당 부분은 아래와 같이 진행된다.

1. Feature Extraction
   + Layer: Backbone (ResNet)
   + Output: Feature map
2. Feature Fusion
   + Layer: FPEM (Feature Pyramid Enhancement Module)
   + Output: Enhanced feature map
3. Detection
   + Layer: Detection Head
   + Output: Text region, text kernel, instance vector
4. Post-processing (Pixel Aggregation, PA)
   + Output: Axis of bbox (bounding box)

## Goal

+ FPEM의 stack 수 편집
  + 원문 코드: 2 stacked FPEMs 사용
  + 목표: 4 stacked FPEMs
+ Fine-tuning
  + 목표: 추가된 2 stacked FPEMs 계층만을 훈련

<!-- More -->

---

# Changing PyTorch Model Structure

모델 구조를 변경하기 위해서는 목표 모델이 어떻게 구성되어있는지 파악해야한다.

```python pan_pp.pytorch/models
├── __init__.py
├── backbone
│   ├── __init__.py
│   ├── builder.py
│   └── resnet.py
├── builder.py
├── head
│   ├── __init__.py
│   ├── builder.py
│   ├── pa_head.py
│   ├── pan_pp_det_head.py
│   ├── pan_pp_rec_head.py
│   └── psenet_head.py
├── loss
│   ├── __init__.py
│   ├── acc.py
│   ├── builder.py
│   ├── dice_loss.py
│   ├── emb_loss_v1.py
│   ├── emb_loss_v2.py
│   ├── iou.py
│   └── ohem.py
├── neck
│   ├── __init__.py
│   ├── builder.py
│   ├── fpem_v1.py
│   ├── fpem_v2.py
│   └── fpn.py
├── pan.py
├── pan_pp.py # Here
├── post_processing
│   ├── __init__.py
│   ├── beam_search
│   │   ├── __init__.py
│   │   ├── beam_search.py
│   │   └── topk.py
│   ├── pa
│   │   ├── __init__.py
│   │   ├── pa.cpp
│   │   ├── pa.pyx
│   │   ├── readme.txt
│   │   └── setup.py
│   └── pse
│       ├── __init__.py
│       ├── pse.cpp
│       ├── pse.pyx
│       ├── readme.txt
│       └── setup.py
├── psenet.py
└── utils
    ├── __init__.py
    ├── conv_bn_relu.py
    ├── coordconv.py
    └── fuse_conv_bn.py
```

여기서 모델 구조 변경을 위해 수정할 코드는 `models/neck/fpem_v2.py`가 아니라 `models/pan_pp.py`이다.

```python config.py
model = dict(
    type='PAN_PP',
...
```

왜냐하면 모델을 빌드할 때 `config.py` 파일의 [`type='PAN_PP'`](https://github.com/whai362/pan_pp.pytorch/blob/master/models/__init__.py) 옵션을 통해 `pan_pp.py`로 계층이 구성되기 때문이다.
만약 FPEM 내부 구조를 수정하려한다면 `models/neck/fpem_v2.py`의 코드를 수정해야할 것이다.
해당 코드에서 FPEMs의 계층 수를 변경하기 위해 해당 코드 내에서 FPEMs 계층 정의 부분을 살펴보겠다.

```python pan_pp.py
class PAN_PP(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(PAN_PP, self).__init__()
...
        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)
...
    def forward(self,
...
        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)
```

[`models/neck/builder.py`](https://github.com/whai362/pan_pp.pytorch/blob/master/models/neck/builder.py)에서 정의한 `build_neck()`은 입력된 `config.py` 파일에 맞춰 모듈을 빌드해주는 함수다.
위 코드에서 알 수 있듯, `torch.nn.Module`을 상속받은 `PAN_PP` 객체 내부에 2 stacked FPEMs를 생성자 (`__init__()`)에서 선언해주었다.
이후 모델 학습 및 출력을 위해 `forward()` 메서드에 입력과 출력에 맞게 정의해주었다.
따라서 해당 계층들을 추가하기 위해 아래와 같이 수정할 수 있다.

```python pan_pp.py
class PAN_PP(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(PAN_PP, self).__init__()
...
        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)
        self.fpem3 = build_neck(neck)
        self.fpem4 = build_neck(neck)
...
    def forward(self,
...
        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem3(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem4(f1, f2, f3, f4)
```

FPEM 계층이 입력과 출력의 `shape`가 같은 특징이 있어 이렇게 쉽게 모델 구조를 편집할 수 있다.
이렇게 4 stacked FPEMs를 완성했으니 모델을 빌드하여 계층이 잘 생성되었는지 확인해보겠다.

```python
from mmcv import Config
cfg = Config.fromfile('cfg.py')
from models import build_model
model = build_model(cfg.model)
import torch
model = torch.nn.DataParallel(model).cuda()
print(model)

DataParallel(
  (module): PAN_PP(
...
    (fpem1): FPEM_v2(
...
    )
    (fpem2): FPEM_v2(
...
    )
    (fpem3): FPEM_v2(
...
    )
    (fpem4): FPEM_v2(
...

```

잘 빌드되었으니 해당 계층에 가중치를 생성하기 위해 훈련을 진행해야한다.

---

# Training Only Some Layers

본 절에서는 두 가지를 가정하고 진행한다.

+ 기존 모델 (2 stacked FPEMs)의 pretrained 가중치가 존재
+ 새로 생성한 FPEM 계층 (`PAN_PP.fpem3`, `PAN_PP.fpem4`)의 가중치 X

따라서 가중치가 존재하는 레이어는 훈련하지 않고, 훈련되지 않은 레이어인 `PAN_PP.fpem3`, `PAN_PP.fpem4` 계층만을 훈련하는 방법을 기술하겠다.

```python train.py
def main(args):
    if hasattr(cfg.train_cfg, 'pretrain'):
...
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
...
```

기존 코드에서는 `cfg.train_cfg.pretrain`에 저장되어있는 가중치를 `torch.nn.Module.load_state_dict()` 메서드로 불러와 fine-tuning을 시작하게 된다.
하지만 현재 저 가중치 (`checkpoint['state_dict']`)는 `PAN_PP.fpem3`, `PAN_PP.fpem4`의 정보가 일체 존재하지 않기 때문에 위 코드를 그대로 실행하면 오류가 발생하게 된다.
따라서 `model.load_state_dict(checkpoint['state_dict'], False)`와 같이 실행해야한다.
이렇게 추가한 계층을 제외한 가중치를 모델에 로드했다면 훈련 시 추가한 계층을 제외한 계층의 훈련을 정지시켜야한다.
이를 위해 `torch.nn.Module.named_parameters()`를 활용할 수 있다.
`for n, p in model.named_parameters():`와 같이 사용하며 `n`은 계층의 이름을, `p`는 계층의 파라미터 ([`torch.nn.parameter.Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter))를 의미한다.
[`torch.nn.parameter.Parameter.requires_grad`](https://pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad)을 `False`로 정의하면 gradient 계산 진행이 되지 않아 학습을 제외할 수 있고, `True`로 정의하면 gradient 계산 진행이 되어 학습을 포함할 수 있다.
따라서 반복문과 조건문을 적절히 사용해 아래와 같이 훈련 코드를 사용하면 원하는 계층만을 훈련하고, 나머지 계층은 훈련에서 제외할 수 있다.

```python tarin.py
...
def main(args):
    if hasattr(cfg.train_cfg, 'pretrain'):
...
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        nmd = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in nmd}
        model.load_state_dict(pretrained_dict, False)
        for n, p in model.named_parameters():
            print(n, p.requires_grad)
            if 'fpem3' in n or 'fpem4' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
            print(n, p.requires_grad)
...
```