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
<img width="1359" alt="Results of vanila montage DCGAN according to activation function" src="https://user-images.githubusercontent.com/42334717/183442489-1654cd4e-a9aa-48c6-8f2b-724639b94c9f.png">

+ 생성기를 통해 생성된 이미지의 수준 저조
+ 모드 붕괴 발생

<!-- More -->

***

# Increase Epoch & Seed

> Results of reinforced montage DCGAN by changing epoch and seed
<img width="1941" alt="Results of reinforced montage DCGAN by changing epoch and seed" src="https://user-images.githubusercontent.com/42334717/183445385-156a9556-c801-450d-a1a2-28607adbd51f.png">

+ 생성 이미지의 질 개선 완료
+ 모드 붕괴 개선 필요

***

# Strong Discriminator

> Loss of DCGAN training process to identify the cause of mode collapse
<img width="2654" alt="Loss of DCGAN training process to identify the cause of mode collapse" src="https://user-images.githubusercontent.com/42334717/183453032-4b29d8c7-84a1-499e-bb3c-a6de49799fb8.png">

+ 모드 붕괴의 원인을 생성기에 비해 상대적으로 너무 강력한 판별기로 선정
+ 6th Trial에서 판별기와 생성기의 균형 관측

> Results of reinforced montage DCGAN by editing activation function and convolution layer of discriminator
<img width="1911" alt="Results of reinforced montage DCGAN by editing activation function and convolution layer of discriminator" src="https://user-images.githubusercontent.com/42334717/183453163-905a3223-e7b4-4a4b-83a5-31d4d2da6a63.png">

+ 판별기와 생성기의 loss 균형 != 양질의 생성기

***

# Weak Generator

> Results of reinforced montage DCGAN by editing convolution layer of generator
<img width="1862" alt="Results of reinforced montage DCGAN by editing convolution layer of generator" src="https://user-images.githubusercontent.com/42334717/183460016-69a9d79a-d0b2-433f-94b7-535c2c45cbe6.png">

> Results of reinforced montage DCGAN by changing learning rate of generator
<img width="1861" alt="Results of reinforced montage DCGAN by changing learning rate of generator" src="https://user-images.githubusercontent.com/42334717/183455953-4981b4f9-fbe2-4a5b-81ca-086cbd237075.png">

> Loss of DCGAN training process according to trial
<img width="3448" alt="image" src="https://user-images.githubusercontent.com/42334717/183457229-00744bc3-b260-4f49-b266-10eead5baecc.png">

+ 8th Trial에서 모드 붕괴 개선
+ 최적화를 위해 생성기의 learning rate, `lr` 조절

***

# Strong Discriminator

> Results of reinforced montage DCGAN by editing activation function of discriminator
<img width="1361" alt="Results of reinforced montage DCGAN by editing activation function of discriminator" src="https://user-images.githubusercontent.com/42334717/183459395-884f1dd4-8406-42d5-af93-75e0b1260f6f.png">

+ Activation function으로 `GELU`가 판별기에 부정적 영향을 주어 강력한 판별기를 약화시키기 위해 사용
+ 하지만 너무 큰 성능 저하로 인해 생성기 발산

***

# Weak Generator

> Results of reinforced montage DCGAN by editing learning rate and kernel size of generator
<img width="1941" alt="Results of reinforced montage DCGAN by editing learning rate and kernel size of generator" src="https://user-images.githubusercontent.com/42334717/183594896-44f23a06-c795-4d2f-b27d-f953f8fe3b90.png">

+ Activation function of discriminator: `GELU`
+ Activation function of generator: `GELU` $\rightarrow$ `LeakyReLU`
+ 모드 붕괴 개선 불가