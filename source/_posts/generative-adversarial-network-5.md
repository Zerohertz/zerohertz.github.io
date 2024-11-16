---
title: Generative Adversarial Network (5)
date: 2022-08-08 23:29:52
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
---
# Vanila Montage DCGAN

> Results of vanila montage DCGAN according to activation function
<img src="/images/generative-adversarial-network-5/results-of-vanila-montage-dcgan-according-to-activation-function.png" alt="results-of-vanila-montage-dcgan-according-to-activation-function" width="1359" />

+ 생성기를 통해 생성된 이미지의 수준 저조
+ 모드 붕괴 발생

<!-- More -->

***

# Increase Epoch & Seed

> Results of reinforced montage DCGAN by changing epoch and seed
<img src="/images/generative-adversarial-network-5/results-of-reinforced-montage-dcgan-by-changing-epoch-and-seed.png" alt="results-of-reinforced-montage-dcgan-by-changing-epoch-and-seed" width="1941" />

+ 생성 이미지의 질 개선 완료
+ 모드 붕괴 개선 필요

***

# Strong Discriminator

> Loss of DCGAN training process to identify the cause of mode collapse
<img src="/images/generative-adversarial-network-5/loss-of-dcgan-training-process-to-identify-the-cause-of-mode-collapse.png" alt="loss-of-dcgan-training-process-to-identify-the-cause-of-mode-collapse" width="2654" />

+ 모드 붕괴의 원인을 생성기에 비해 상대적으로 너무 강력한 판별기로 선정
+ 6th Trial에서 판별기와 생성기의 균형 관측

> Results of reinforced montage DCGAN by editing activation function and convolution layer of discriminator
<img src="/images/generative-adversarial-network-5/results-of-reinforced-montage-dcgan-by-editing-activation-function-and-convolution-layer-of-discriminator.png" alt="results-of-reinforced-montage-dcgan-by-editing-activation-function-and-convolution-layer-of-discriminator" width="1911" />

+ 판별기와 생성기의 loss 균형 != 양질의 생성기

***

# Weak Generator

> Results of reinforced montage DCGAN by editing convolution layer of generator
<img src="/images/generative-adversarial-network-5/results-of-reinforced-montage-dcgan-by-editing-convolution-layer-of-generator.png" alt="results-of-reinforced-montage-dcgan-by-editing-convolution-layer-of-generator" width="1862" />

> Results of reinforced montage DCGAN by changing learning rate of generator
<img src="/images/generative-adversarial-network-5/results-of-reinforced-montage-dcgan-by-changing-learning-rate-of-generator.png" alt="results-of-reinforced-montage-dcgan-by-changing-learning-rate-of-generator" width="1861" />

> Loss of DCGAN training process according to trial
<img src="/images/generative-adversarial-network-5/loss-of-dcgan-training-process-according-to-trial.png" alt="loss-of-dcgan-training-process-according-to-trial" width="3448" />

+ 8th Trial에서 모드 붕괴 개선
+ 최적화를 위해 생성기의 learning rate, `lr` 조절

***

# Strong Discriminator

> Results of reinforced montage DCGAN by editing activation function of discriminator
<img src="/images/generative-adversarial-network-5/results-of-reinforced-montage-dcgan-by-editing-activation-function-of-discriminator.png" alt="results-of-reinforced-montage-dcgan-by-editing-activation-function-of-discriminator" width="1361" />

+ Activation function으로 `GELU`가 판별기에 부정적 영향을 주어 강력한 판별기를 약화시키기 위해 사용
+ 하지만 너무 큰 성능 저하로 인해 생성기 발산

***

# Weak Generator

> Results of reinforced montage DCGAN by editing learning rate and kernel size of generator
<img src="/images/generative-adversarial-network-5/results-of-reinforced-montage-dcgan-by-editing-learning-rate-and-kernel-size-of-generator.png" alt="results-of-reinforced-montage-dcgan-by-editing-learning-rate-and-kernel-size-of-generator" width="1941" />

+ Activation function of discriminator: `GELU`
+ Activation function of generator: `GELU` $\rightarrow$ `LeakyReLU`
+ 모드 붕괴 개선 불가