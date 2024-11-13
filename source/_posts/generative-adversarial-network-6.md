---
title: Generative Adversarial Network (6)
date: 2022-08-14 02:03:59
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# CycleGAN

+ pix2pix
  + Self-supervised
  + Loss: Minimize the difference between output $G(x)$ and ground truth $y$
    + $\underset{(x,y)}{\Sigma}||y-G(x)||_1$
  + Ex) 흑백 $\rightarrow$ 컬러
+ GAN
  + Loss: Another deep network point out the difference
    + $\arg\underset{G}{\min}\underset{D}{\max}\mathbb{E}_{x,y}[\log{D(G(x))}+\log(1-D(y))]$
  + $D$ tries to identify the fakes
  + $G$ tries to synthesize fake images that fool $D$
 
<!-- More -->

+ CycleGAN: 항상 pair dataset을 가지지 못하기 때문에 이용
  + Loss: $G(x)$ should just look photorealistic and $F(G(x))$ should be $F(G(x))=x$, where $F$ is the inverse deep network
    + $L_{GAN}(G(x),y)+||F(G(x))-x||_1$
    + $L_{GAN}(F(x),y)+||G(F(x))-x||_1$
  + GANs with cross-entropy loss
    + $L_{GAN}(G,D_Y,X,Y)=\mathbb{E}\_{y\sim p_{data}(y)}[\log{D_Y(y)}]+\mathbb{E}\_{x\sim p_{data}(x)}[\log{(1-D_Y(G(x)))}]$
  + Least square GANs
    + $L_{LSGAN}(G,D_Y,X,Y)=\mathbb{E}\_{y\sim p_{data}(y)}[(D_Y(y)-1)^2]+\mathbb{E}\_{x\sim p_{data}(x)}[D_Y(G(x))^2]$
    + Vanishing gradient problem 개선 $\rightarrow$ stable training, better results

***

# Style Transfer

+ Style Transfer: 두 영상 (content image, style image)가 주어졌을 때 이미지의 주된 형태는 content image와 유사하게 유지하며 스타일만을 style image와 유사하게 변환

<div style="overflow: auto;">

> Content Loss
$$
L_{content}(\vec{p},\vec{x},l)=\frac{1}{2}\underset{i,j}{\Sigma}(F_{ij}^l-P_{ij}^l)^2
$$
</div>

+ $\vec{p}$: content image
+ $\vec{x}$: input image
+ $l$: layer
+ $P^l_{ij}$: the activation $i^{th}$ filter at position $j$ in layer $l$ (content image $\vec{p}$)
+ $F^l_{ij}$: the activation $i^{th}$ filter at position $j$ in layer $l$ (input image $\vec{x}$)

> Style Loss
$$
L_{style}(\vec{a},\vec{x})=\overset{L}{\underset{l=0}{\Sigma}}w_lE_l
$$

+ $\vec{a}$: style image
+ $A^l_{ij}$: the inner product between $F_i^l$ and $F_j^l$ in layer $l$ (style image $\vec{a}$)
+ $G^l_{ij}$: the inner product between $F_i^l$ and $F_j^l$ in layer $l$ (input image $\vec{x}$)

<div style="overflow: auto;">

> Total Loss
$$
L_{total}(\vec{p},\vec{a},\vec{x})=\alpha L_{content}(\vec{p},\vec{x})+\beta L_{style}(\vec{a},\vec{x})
$$
</div>