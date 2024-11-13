---
title: 'Paper Review: DEER'
date: 2023-05-24 23:14:40
categories:
- 5. Machine Learning
tags:
- Paper Review
---
> [DEER: Detection-agnostic End-to-End Recognizer for Scene Text Spotting](https://arxiv.org/pdf/2203.05122.pdf)

# Introduction

이 논문은 NAVER Clova 팀에서 개발한 end-to-end text spotting 모델을 제안한다.
End-to-end text spotting 모델은 text detector와 recognizer로 구성되어있다.
해당 논문의 저자들은 기존 text spotting 모델들은 detector와 recognizer 사이에 다소 밀접하게 결합되어있다고 아래와 같이 주장한다.

1. Detector에서 잘라낸 이미지는 recognizer에 입력되기 때문에 필연적으로 detector의 성능에 따라 recognizer의 성능이 결정된다.
2. Detctor에서 crop된 영역의 localized features를 recognzier에 전달하여 detector의 종속성을 감소시킬 수 있지만 여전히 존재하는 detector의 오류 누적으로 인해 recognition 실패가 발생할 수 있다.
3. Feature pooling과 masking은 end-to-end text spotting 모델을 학습하기 위해 bounding boxes 데이터가 여전히 필요하다.

이에 대한 자세한 설명은 [발표자료](https://deview.kr/data/deview/session/attach/%5B143%5D%EA%B2%80%EC%B6%9C%EA%B3%BC%EC%9D%B8%EC%8B%9D%EB%AA%A8%EB%8D%B8%EC%9D%84%ED%95%98%EB%82%98%EB%A1%9C+challenge+%EC%9A%B0%EC%8A%B9+OCR%EB%AA%A8%EB%8D%B8+%EC%83%88%EC%B6%9C%EC%8B%9C.pdf)의 6, 7, 8페이지에서 확인할 수 있다.

Object detection 분야에서는 end-to-end Transformer 기반 접근 방식이 발전함에 따라 이미지의 개별 객체를 엄격하게 recognize하기 위해 정확한 영역 정보, 정교한 ground truth 할당 및 feature pooling이 필요하지 않다는 것이 분명해지고 있다.
그렇기에 저자들은 detection 결과의 정확성에 대한 의존성을 완화할 수 있는 DEER (Detection-agnostic end-to-end Recognizer)를 제안한다.
DEER는 정확한 text 영역을 추출하기 위해 detector에 의존하지 않고 detector가 각 text instance에 대한 single reference point를 localize하도록 한다.
Reference point 주위의 text를 포괄적으로 recognize하는 text decoder는 text sequence를 decoding하는 동안 특정 text instance의 attending region을 결정하는 방법을 학습한다.
DEER는 기존 모델들과 다르게 detecotr의 역할이 single reference point를 localize하는 것 뿐이기 때문에 훨씬 더 다양한 검출 알고리즘과 주석을 사용할 수 있다.
이러한 접근 방식을 통해 pooling operations와 polygon-type 주석 없이 회전 및 곡선 text instance를 자연스럽게 처리할 수 있다.

<!-- More -->

---

# Related Works

저자들은 다양한 end-to-end text spotting 모델들과 제안된 모델을 개발할 때 영감을 받은 최신 object detection과 segmentation 모델을 리뷰했다.

## End-to-end Scene Text Spotting

초기의 end-to-end text spotting 모델 중 TextBoxes++는 text detector와 recognizer를 각각 학습하고 이후에 이들을 결합하였다.
하지만 후속 연구들에 따르면 이러한 end-to-end 모델들은 detection 결과에 크게 의존하기 때문에 detector와 recognizer를 함께 학습하면 최종 성능이 향상된다는 것이 밝혀졌다.
그리고 ABCNet, TextDragon, MaskTextSpotterV3, MANGO와 같은 모델들은 앞서 서술한 것과 같이 최종 성능이 detection 결과에 크게 의존하는 문제 때문에 정교한 detection 및 feature pooling/masking 알고리즘에 대해 연구했다.
반면 저자들이 제안하는 방법은 detector가 recognizer에 reference point만을 제공하는 점이 다르며 recognzier는 reference point와 전체 입력을 통해 출력 text sequence를 decoding한다.

## End-to-end Object Detection and Segmentation

Transformer 기반 object detection과 instance segmentation 모델들은 아래와 같이 매우 활발히 연구되고 있고, 획기적인 결과들을 달성했다.

+ DETR: Spatial anchor-boxes 및 non-maximum suppresion과 같은 정교한 수작업 구성 요소를 사용하지 않고 경쟁적인 결과를 달성했다.
+ Deformable DETR: Deformable attention을 사용하여 DETR의 학습 속도를 개선하고 multi-scale features를 사용하여 작은 물체에 대한 성능 문제를 해결했다.
+ Efficient DETR: Well-initialized reference points와 초기의 dense object detection 단계에서 생성된 object queries를 사용하여 학습 속도를 더욱 개선했다.

Panoptic segmentation에 대한 연구들 중 아래의 연구를 리뷰했다.

+ Panoptic SegFormer: Two-stage decoders (location decoder, mask decoder)를 사용하여 state-of-the-art를 달성했다.

위의 연구들을 통해 명시적인 detetion proposal은 높은 recognition 성능을 위해 필수적이지 않음을 알 수 있다.
이러한 결론을 통해 저자들은 Transformer 구조와 reference points (혹은 location queries)의 개념을 채택하여 detector와 recognizer의 의존성을 완화했다.

---

# DEER

DEER는 backbone, Transformer encoder, location head, text decoder로 구성된다.
각 파트의 역할은 아래와 같다.

+ Transformer encoder: Backbone에서 생성된 multi-scale feature maps를 결합
+ Location head: Text instances와 bounding boxes의 reference points를 예측
+ Text decoder: Reference points로 지정된 각 text instance에서 character sequences를 생성

전체 모델의 순전파는 아래와 같이 진행된다.

1. Backbone
   + 입력: $X$ (input image)
     + $X\in\mathbb{R}^{H\times W\times 3}$
   + 출력: $C_2$, $C_3$, $C_4$, $C_5$ (feature maps)
     + $C_2\in\mathbb{R}^{H/4\times W/4}$
     + $C_3\in\mathbb{R}^{H/8\times W/8}$
     + $C_4\in\mathbb{R}^{H/16\times W/16}$
     + $C_5\in\mathbb{R}^{H/32\times W/32}$
2. Fully-connected layer, group normalization (project into 256 channels)
   + 입력: $C_2$, $C_3$, $C_4$, $C_5$ (feature maps)
   + 출력: $T$ (feature tokens)
     + $T\in\mathbb{R}^{(L_2+L_3+L_4+L_5)\times 256}$
     + $L_i$ = flattened length of $C_i$ = $\frac{H}{2^i}\times\frac{W}{2^i}$
3. Transformer encoder
   + 입력: $T$ (feature tokens)
   + 출력: $F$ (refined features)
4. Location head
   + 입력: $F$ (refined features) of size $L_2$ (corresponds to $C_2$)
   + 출력: Reference points
5. Text decoder
   + 입력: Reference points, $F$ (refined features)
   + 출력: Chracter sequences

## Transformer Encoder

높은 해상도의 multi-scale features는 text recognition의 성능 향상을 가능케 하지만 self-attentioon의 연산 비용이 입력 길이에 따라 제곱으로 증가하기 때문에 multi-scale features의 연결에 Transformer를 사용하는 것은 비효율적이다.
따라서 기존의 Transformer encoders는 $C_5$와 같은 저해상도 features를 사용했었다.
하지만 본 연구에서 저자들은 입력 길이에 따라 선형으로 확장되는 deformable attention을 사용하였고 이러한 효율성으로 DEER의 encoder는 고해상도 features를 정제하여 multi-scale feature tokens $F$를 생성할 수 있다.

<div style="overflow: auto;">

> Deformable Attention: 효율성과 위치 인식으로 인해 encoder와 decoder 모두에 중요한 구성 요소
> $$DeformAttn_h(A_{hqk},p_{\mathrm{ref}},\Delta p_{hqk})=W^o_h[\Sigma^K_{k=1}A_{hqk}\cdot W^k_hx(v,p_{\mathrm{ref}}+\Delta p_{hqk})]$$
</div>

+ $x(v,p)$: 위치 $p$의 value features $v$에서 features를 추출하는 bilinear interpolation
+ $W^o_h\in\mathbb{R}^{C\times C_m}$, $W^k_h\in\mathbb{R}^{C_m\times C}$: Linear projection
+ $p_{\mathrm{ref}}$: Reference points
  + Query features에 linear projections 적용
  + Encoder에서 $[0,1]\times[0,1]$로 정규화된 좌표와 함께 고정된 reference points 사용
  + [Efficient DETR](https://arxiv.org/abs/2104.01318)와 같이 미리 정의된 reference points 대신 이미지에 따른 reference points를 사용하여 모델의 학습을 가속화
  + 각 object에 위치한 reference points를 이용하여 특정 text instances를 decoding
+ $\Delta p_{hqk}$: Sampling offsets
  + Query features에 linear projections 적용
+ $A_{hqk}$: Attention weights
  + Query features에 linear projections 적용
  + Softmax 적용

## Location Head

[Panoptic SegFormer](https://arxiv.org/abs/2109.03814), [Efficient DETR](https://arxiv.org/abs/2104.01318)의 접근 방식을 통해 location head를 사용하여 text decoder의 reference points (i.e., text instance의 중심 위치)를 예측하고 이러한 정보를 토대로 물체를 recognize하고 구별하는데 도움을 준다.
또한 evaluation metrics를 계산하기 위해 text instances의 bounding polygon을 추출하기 위한 segmentation map을 제공한다.
저자들은 [differentiable binarization (DB)](https://arxiv.org/abs/1911.08947)을 채택하여 평가에 필요한 text instances의 bounding polygon을 추출하였다.
구체적으로는 아래와 같이 진행된다.

1. $F$에서 $C_2$에 해당하는 $L_2$ 크기의 feature tokens를 추출하여 $(H/4,W/4)$로 reshape한다.
2. Transposed convolution, group normalization, relu로 구성된 separated segmentation head로 binary map과 threshold map을 출력한다.
3. Inference 단계에서는 감지된 bounding polygons에서 text instances의 중심 좌표를 reference points로 사용한다.

## Text Decoder

Transformer decoder로 구성된 recognition branch는 text instance 내 character sequences를 autoregressively로 예측한다.

+ $Q$: Text decoder에 대한 query로 character embedding, positional embedding, reference point $p_{\mathrm{ref}}$로 구성
+ $K$: Text decoder의 keys $\leftarrow$ Transformer encoder의 출력인 feature tokens $F$
+ $V$: Text decoder의 values $\leftarrow$ Transformer encoder의 출력인 feature tokens $F$

Self-attention, deformable attention, feed-forward layers를 통해 queries를 전달한다.
[Panoptic SegFormer](https://arxiv.org/abs/2109.03814), [Spatial Attention Mechanism](https://arxiv.org/abs/1904.05873)으로부터 영감을 받아 저자들은 $F$에 deformable attention 대신 regular cross attention을 교대로 추가하였다.
학습 단계 동안 $N_t$ text boxes가 샘플링되어 계산된 중심 좌표가 decoder의 reference points로 사용되며 이를 통해 location head와 text decoder의 독립적 학습이 가능하다.
학습 단계에서는 ground truth regions에서 계산한 points를 text decoder의 reference points으로 사용
평가 단계에서는 detection branch의 중심 좌표가 reference point로 사용된다.
Ground truth (학습)와 model prediction (평가)의 차이를 줄이기 위해 아래 방정식을 사용해 학습 단계에서 중심 좌표가 교란된다.

<div style="overflow: auto;">

$$
p_{\mathrm{ref}}=p_c+\frac{\eta}{2}\min{(||p_{tl}-p_{tr}||,||p_{tl}-p_{bl}||)}, \\
\eta\sim \mathrm{Uniform}(-1,1)
$$
</div>

+ $p_c$: Ground truth polygon의 중심
+ $p_{tl}$: Top-left point의 좌표
+ $p_{tr}$: Top-right point의 좌표
+ $p_{bl}$: Bottom-left point의 좌표

Inference 단계에서는 detection 단계에서 추출된 text regions의 중심점을 reference point로 사용한다.

## Optimization

$$
L = L_r + \lambda_s L_s + \lambda_b L_b + \lambda_t L_t
$$

+ $L$: Loss function
+ $L_r$: Autogressive text recognition loss
  + Character sequences의 예측된 가능성과 text box에 따른 ground truth text label 사이의 softmax cross entropy로 계산
+ $L_s,\ L_b,\ L_t$: Losses from differentiable binarization
  + $L_s$: Loss for the probability map
    + Binary cross entropy with hard negative mining
  + $L_b$: Loss for the binary map
    + Dice Loss
  + $L_t$: Loss for the threshold map
    + $L_1$ distance loss

DEER는 추론 시 location head의 probability 만을 사용한다.
Probability map은 특정된 threshold에 의해 이진화되어 있고, 연결된 component들은 binary map에 의해 추출된다.
추출된 영역의 size는 실제 text region보다 작으므로 Vatti clipping algorithm을 사용한다. ($D=\frac{A\times r}{L}$)
이렇게 확장된 각 영역에서 polygon을 추출하고 중심 좌표를 계산하여 decoder에 reference point로 전달한다.
마지막으로, decoder은 해당 text region의 character sequences들을 greedily predict한다.

---

# Reference

+ [검출과 인식 모델을 하나로?: challenge 우승 OCR 서비스 모델 새 출시!](https://deview.kr/data/deview/session/attach/%5B143%5D%EA%B2%80%EC%B6%9C%EA%B3%BC%EC%9D%B8%EC%8B%9D%EB%AA%A8%EB%8D%B8%EC%9D%84%ED%95%98%EB%82%98%EB%A1%9C+challenge+%EC%9A%B0%EC%8A%B9+OCR%EB%AA%A8%EB%8D%B8+%EC%83%88%EC%B6%9C%EC%8B%9C.pdf)

---

그런데 논문에서 $p_{\mathrm{ref}}$, $q_{\mathrm{ref}}$ 이렇게 두 변수가 나오는데,, 오타일까요?