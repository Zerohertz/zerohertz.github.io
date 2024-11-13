---
title: 'Paper Review: Geometric Quality Prognosis in Roll-to-Roll Slot-die Coating Process'
date: 2022-02-16 14:22:48
categories:
- Etc.
tags:
- Paper Review
- Mechanical Engineering
---
# Lee et al., 2019

본 논문에서는 $th_{m,w}$와 $th_{e,w}$의 비교를 통해 롤투롤 슬롯 다이 코팅에서의 소재와 잉크 사이의 계면에서 잉크의 역학을 분석하고 잉크 점도, 유속 및 코팅 갭에 따른 안정적 공정 조건을 정의했다.

+ $th_{m,w}$: The minimum permissible coated layer thickness
+ $th_{i,w}$ The ideal wet thickness of the coated layer
+ $th_{e,w}$ The estimated wet thickness in the steady state
+ $th_{e,d}$ The estimated thickness of the coated layer after drying

<!-- More -->

## $th_{m,w}$ 유도

1. $DMC$ (Downstream Meniscus Curvature)는 잉크의 표면 장력 ($\sigma$)을 잉크의 압력과 대기압의 차이 ($\Delta{P}$)로 나눈 값이다.
   + $DMC=\frac{\sigma}{P_{ink}-P_{air}}$
2. $DMC$가 최소인 경우, 압력 구배 $\Delta{P}$는 최대가 된다.
   + $DMC=\frac{G-th_{m,w}}{2}$
     + $G$: Downstream meniscus 곡률이 $DMC$로 메니스커스가 형성되는 코팅 갭
3. Ruschak's model을 이용하여 코팅층 두께, 잉크의 표면 장력과 점도에 따른 압력 구배를 구하고 낮은 Capillary number ($Ca\leq0.1$)에서 Laudau-Levich 식을 적용하면 아래와 같다.
   + $\Delta{P}=1.34Ca^{\frac{2}{3}}\frac{\sigma}{th_{m,w}}$
     + $Ca=\frac{\mu V}{\sigma}$ ($\mu$: Viscosity of ink, $V$: Web speed)
4. 위의 식들을 조합하여 정리하면 최소 습윤 두께를 구할 수 있다.
   + $th_{m,w}=\frac{G}{(\frac{2}{1.34Ca^{2/3}}+1)}$

## $th_{i,w},th_{e,w}$ 유도


1. 질량 보존의 법칙에 의해 아래와 같이 정의할 수 있다.
   + $\dot{m}_{cv}=\frac{d}{dt}\int \rho f(x,t)=\rho (f\_r-nd(th\_{i,w}V))$
     + $\dot{m}_{cv}$: Control volume 내의 질량 변화
     + $\rho$: 잉크의 밀도
     + $f(x,t)$: Control volume 내의 부피 변화
     + $f_r$: 잉크 토출 유량
     + $n$: 스트립의 수
     + $d$: 코팅 폭
   + $th_{i,w}=\frac{f_r}{ndV}$: 이상적인 습윤 두께 ($\because\dot{m}_{cv}=0$)
   + $th_{i,w}>th_{m,w}$: Stable
2. 실험적 계수 $K$를 통해 습윤 두께를 추정할 수 있다.
   + $th_{e,w}=K(th_{i,w})=\frac{Kf_r}{ndV}$
     + $th_{e,w}$: 추정 습윤 두께

## $th_{e,d}$ 유도

+ $th_{e,d}=w(th_{e,w})=\frac{wKf_r}{ndv}$
  + $th_{e,d}$: 건조 후의 추정 두께
  + $w$: Solute의 weight percent

***

# Reference

+ [Large area electrolyte coating through surface and interface engineering in roll-to-roll slot-die coating process](https://www.sciencedirect.com/science/article/pii/S1226086X19301728)