---
title: 'Paper Review: GANILLA'
date: 2022-08-21 16:24:40
categories:
- 5. Machine Learning
tags:
- Paper Review
---
> [Hicsonmez, Samet, et al. "GANILLA: Generative adversarial networks for image to illustration translation." Image and Vision Computing 95 (2020): 103886.](https://arxiv.org/pdf/2002.05638.pdf)

# Introduction

+ 생산적 적대 신경망 (Generative Adversarial Network, GAN) 기반 아동도서의 삽화 (illustration) 스타일 이미지 변환
  + 일반적인 그림과 만화와 다르게 사물이 포함되지만 추상화 수준이 매우 높음
  + 기존 모델 (CycleGAN, DualGAN)을 통해 추상화 스타일과 삽화의 내용 간 균형을 다루는데 한계점 존재
+ Goal: 주어진 일러스트 작가의 스타일을 전달하며 주어진 이미지의 콘텐츠를 보존하는 생성기 개발
  + 정렬되지 않은 두 개의 개별 image dataset $\rightarrow$ Unpaired approach
    + Source domain (natural images)
    + Target (illustrations)
  + 스타일과 콘텐츠의 불균형 문제
    + Residual layer에서 특징 맵을 다운 샘플링하여 새로운 생성기 네트워크 제시
    + 콘텐츠를 더 잘 전송하기 위해 skip connection 및 upsampling 사용 $\rightarrow$ 낮은 수준의 feature을 높은 수준의 feature와 병합
+ Unpaired style transfer approach의 evaluation
  + 일반적으로 image-to-image translation 모델의 평가는 정성적
  + 생성된 이미지에 대해 짝지어진 ground-truth가 존재하지 않아 직접적으로 정량적 평가 불가
  + 콘텐츠 및 스타일 분류기 기반 정량적 평가 프레임워크 제안
+ Highlights
  + Image-to-image style and content transfer의 새로운 연구
  + 24명의 아티스트에 대한 약 9500개의 illustration으로 구성된 dataset 제공
  + 스타일과 콘텐츠의 균형이 맞는 새로운 generator network 제안
  + 콘텐츠와 스타일 측면에서 이미지 생성 모델의 새로운 정량적 평가 프레임워크 제안

<!-- More -->

***

# GANILLA

+ Preliminary experiments: image-to-illustration translation에 대해 기존 모델인 쌍을 이루지 않은 image-to-image translation 모델이 스타일과 콘텐츠를 동시에 전송하지 못하는 한계점 존재
  + 콘텐츠를 보존하며 스타일을 전달하는 새로운 generator network 제시
  + 2가지 ablation 모델 제시

## Generator

+ 저수준 feature를 사용해 스타일을 전송하며 콘텐츠 보존
+ 다운 샘플링 단계와 업 샘플링 단계, 총 두 단계로 구성
  + 다운 샘플링 단계: 수정된 ResNet-18 네트워크 사용
    + 저수준 feature를 통합하기 위해 다운 샘플링의 각 레이어에서 이전 레이어의 feature 연결
    + 저수준 레이어는 형태적 특징, 가장자리 및 모양과 같은 정보 통합
    + 전송된 이미지가 입력 콘텐츠의 하위 구조를 가지도록 설계
  + 업 샘플링 단계: summation layer에 skip connection을 통해 다운 샘플링 단계에서 각 레이어의 출력을 사용하여 하위 수준 feature를 제공하고 업 샘플링 (Nearest Neighbor)
    + 콘텐츠 보존에 이점 존재

## Discriminator

+ $70\times70$ PatchGAN: image-to-image translation에 성공적으로 사용된 모델

## Training Option

+ Cycle-consistency
  + 첫 번째 세트 ($G$): 소스 이미지를 대상 도메인에 매핑 시도
  + 두 번째 세트 ($F$): 대상 도메인 이미지로 입력 후 순환 방식으로 소스 이미지 생성 시도
+ Loss function: generator 및 discriminator 쌍에 대해 두 가지 손실 사용
  + Minimax loss
  + Cycle consistency loss: 생성된 손실이 소스 도메인에 다시 매핑될 수 있도록 도움 ($L_1$ distance)
+ Dataset
  + 서로 다른 짝을 이루지 않은 image dataset (source domain & target domain)
  + $256\times256$
+ Etc.
  + Learning rate: 0.0002
  + Solver: Adam
  + Epoch: 200

***

# Evaluation

+ 비교 대상 (state-of-the-art GAN methods that use unpaired data)
  + CartoonGAN
  + CycleGAN
  + DualGAN
+ Two main factors which determine the quality of the GAN generated illustrations
  + Style-CNN: Having target style
    + 스타일 전달 측면에서 결과가 얼마나 좋은지 평가
  + Content-CNN: Preserving the content
    + 입력 이미지의 정보 보존 여부 감지

## Quantitative Analysis and User Study

+ GANILLA
  + 고유한 아티스트 스타일로 이미지 생성
  + 약간의 결함 존재
+ CycleGAN
  + 스타일을 잘 전달하지만 기존의 콘텐츠 변형 발생
  + 생성 이미지에 소스 illustration의 얼굴, 사물과 같은 것을 환각
+ CartoonGAN & DualGAN
  + 콘텐츠를 잘 보존하지만 다양한 경우에서 스타일 전달 측면에서 저조

## Quantitative Analysis

+ Style-CNN
  + 스타일별 분류기를 훈련시키기 위해 스타일을 유지하며 시각적 콘텐츠에서 훈련 이미지 분리
  + Illustration 이미지에서 작은 패치 ($100\times100$ pixel)를 무작위로 자르고 해당 패치를 사용하여 스타일 분류기 Style-CNN 훈련
  + Training set: illustration 아티스트를 위한 10개 클래스와 자연 이미지에 대한 1개의 클래스로 구성
  + 분류기를 테스트하기 위해 생성된 이미지만을 사용
+ Content-CNN
  + 콘텐츠 보존을 평가하기 위해 콘텐츠 분류기 Content-CNN 훈련
  + 특정 장면 범주 (숲, 거리, etc.)를 콘텐츠로 정의
  + 특정 스타일로 산 이미지를 생성한다면 생성 이미지 또한 산 이미지로 분류되어야 함

## Ablation Experiments

+ 모델의 효과를 자세히 평가하기 위해 두 가지 절제 실험 수행
  1. 다운 샘플링 부분 (Model 1): 다운 샘플링 CNN을 원본 ResNet-18로 교체하여 수정 효과 확인
  2. 업 샘플링 부분 (Model 2): deconv layer가 존재하는 다운 샘플링 CNN 사용
+ Model 1
  + GANILLA와 유사한 콘텐츠 점수
  + 스타일 점수 저조
  + 기존의 ResNet-18 구조를 수정하여 GANILLA가 입력 이미지를 성공적으로 스타일화할 수 있음을 시사
+ Model 2
  + GANILLA보다 향상된 스타일 점수
  + 매우 저조한 콘텐츠 점수
  + 업 샘플링 부분에서 낮은 수준의 기능을 사용하는 것이 콘텐츠를 보존하는 부분에 큰 도움이 됨을 시사

***

# Conclusion

+ 가장 광범위한 아동도서 illustration dataset과 이미지를 illustration으로 translation하기 위한 새로운 generator network 제시
+ Illustration dataset은 매우 추상적인 대상과 형태를 포함하므로 기존의 generator network는 콘텐츠와 스타일을 동시에 전달하지 못하는 한계점 존재
+ 이를 극복하기 위해 GANILLA는 다운 샘플링 상태와 업 샘플링 부분에서 낮은 수준의 feature 사용
+ Image-to-image translation domain에서 generator 모델을 평가하기 위한 metric이 존재하지 않으므로 해당 문제를 해결하기 위해 평가 프레임워크 제시 $\rightarrow$ 스타일과 콘텐츠 측면을 별도로 측정하는 두 개의 CNN