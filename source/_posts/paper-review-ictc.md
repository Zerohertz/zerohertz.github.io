---
title: 'Paper Review: IC|TC'
date: 2024-02-19 08:24:37
categories:
- 5. Machine Learning
tags:
- Paper Review
---
> [Image Clustering Conditioned on Text Criteria](https://arxiv.org/abs/2310.18297)

# Introduction

Image clustering은 unsupervised learning task으로 연구되어 왔으며 대량의 visual data 구축, label이 없는 image dataset의 labeling 비용 감소, image 검색 system 성능 향상에 사용되었다.
최신의 deep image clustering 방법들은 ground truth로 간주되는 dataset의 사전 정의된 class label으로 종종 평가된다.

하지만 실제로는 각 사용자들이 동일한 dataset에 대해 clustering할 때 서로 다른 기준에 따라 여러 결과를 원할 수 있다.
그럼에도 기존 clustering 방법은 사용자가 clustering 기준을 제어할 수 있는 직접적인 mechanism을 제공하지 않는다. (기존 clustering 방법의 기준은 neural network의 inductive bias, loss function, data augmentation, feature extractor에 의해 결정될 가능성이 높다.)
따라서 저자들은 이러한 한계점을 개선하기 위해 사용자가 지정한 기준을 기반으로 단일 dataset에서 다양한 결과를 가능하게 하고 기존의 암묵적으로 지시된 clustering process를 혁신한다.

<!-- More -->

최근 language and multi-modal foundation model은 전례 없는 수준으로 사람의 지시를 이해하고 따르는 능력으로 상당한 관심을 받고있다.
Large language model (LLM)은 zero-shot 또는 few-shot의 understanding (이해), summarizing (요약), reasoning (추론)과 같은 광범위한 자연어 task를 아주 잘 수행한다.
Vision-language model (VLM)은 시각적 맥락에서 자연어 지시를 해석하고 심층적인 image 분석과 복잡한 추론을 나타내는 것처럼 보이는 응답을 생성한다.

저자들은 자연어 text에서 제공되는 사용자 지정 기준에 따라 image clustering을 수행하기 위한 foundation model 기반 새로운 방법인 Image Clustering Conditioned on Text Criteria (IC|TC)를 제시한다.
사용자는 관련 clustering 기준을 사용하여 방법을 지시하고 동일한 dataset를 여러 다른 기준으로 clustering할 수 있으며 clustering 결과가 만족스럽지 않다면 사용자는 text 기준을 수정하여 clustering 결과를 반복적으로 세분화할 수 있다.
IC|TC는 최소한의 실용적인 사람의 개입을 필요로 하는 대신 clustering 결과에 대한 중요한 제어권을 사용자에게 부여한다.
저자들은 이러한 이유로 기존의 순수 unsupervised clustering 방법에 비해 IC|TC가 더 실용적이고 강력하다고 주장한다.

---

# Task Definition: Image Clustering Conditioned on Iteratively Refined Text Criteria

> Main task: Image들, cluster의 수 $K$, 자연어로 표현된 사용자 지정 기준이 주어지면 image들을 $K$ cluster로 분할하고 각 cluster는 지정된 사용자 기준과 일치하는 방식으로 구별

최근 image clustering 방법은 CIFAR-10과 같은 dataset에 대해 사전 정의된 class label과 일치하는 cluster를 찾는다.
Cluster의 의미는 foreground object의 category에 해당하는 경향이 존재하고 이렇게 cluster가 선택되는 방식의 원인은 neural network의 inductive bias, loss function, data augmentation, feature extractor일 가능성이 높다.
하지만 위와 같은 고전적 clustering 방법에 의해 생성된 cluster는 사용자가 원하는 기준과 일치하지 않을 수 있다.

## Iterative Refinement of Text Criteria

사용자는 text 기준을 지정, clustering을 수행, cluestring 결과 검사를 진행하고 만족하지 않으면 text 기준을 편집하여 clustering 결과를 반복적으로 구체화하여 text 기준을 선택한다.
간혹 사용자가 정의한 text 기준이 원하는 clustering 결과로 이어지는 경우도 존재하지만 그렇지 않은 경우 이러한 반복 prompt engineering 절차는 원하는 결과로 수렴하기 위한 실용적인 수단을 제공한다.
실제로 기존 clustering algorithm의 hyperparameter는 사용자가 clustering 출력을 검사하고 그에 따라 parameter를 조정하는 반복 process를 통해 선택하게 된다.
저자들은 text 기준을 반복적으로 결정하는 process를 명시적으로 인정하고 이를 main task의 일부로 간주한다.

## Comparison with Classical Clustering

순전히 비지도 방식인 고전적 image clustering과 다르게 본 논문의 task는 사용자가 clustering 기준을 지정한다.
Deep clustering 방법은 dataset의 사전 정의된 label에 대해 평가되는 경우가 많으며 이러한 label들은 foreground object의 유형에 초점을 맞추는 경향이 존재한다.
하지만 clustering algorithm이 어떻게 임의의 기준으로 clustering을 수행할 수 있는지 또는 없는지에 대해 질문이 제기되어 여러 연구가 수행되었다.
사용자 정의 text 기준을 사용하면 고전적 비지도 clustering의 instance가 되지 않지만 임의의 기준으로 clustering을 수행하는 것이 목표라면 사용자의 필요이고 실질적 개입이다.

## Comparison with Zero-shot Classification

사전 정의된 class가 필요하고 단순히 image를 이러한 class에 할당하는 zero-shot classification과 cluster를 찾고 image를 cluster에 할당하는 본 논문에서 제시하는 방식은 다르다.
Zero-shot classification은 사용자가 모든 $K$ cluster의 기준을 명시적이고 정확하게 설명할 때 저자들이 언급하는 task의 instance로 간주될 수 있다.

---

# IC|TC: Image Clustering Conditioned on Text Criteria

IC|TC는 선택적 반복 외부 loop가 존재하는 3가지 단계로 구성된다.
$\boldsymbol{\mathrm{TC}}$ (text criterion)는 아래와 같은 text prompt를 통해 3단계로 통합된다.

<div style="overflow: auto;">

$\mathrm{P}\_{\text{step1}}(\boldsymbol{\mathrm{TC}})=\text{"Chracterize the image using a well-detailed description"}+\boldsymbol{\mathrm{TC}}$
$\mathrm{P}\_{\text{step2a}}(\boldsymbol{\mathrm{TC}})=\text{"Given a description of an image, label the image"}+\boldsymbol{\mathrm{TC}}$
$\mathrm{P}\_{\text{step2b}}(\boldsymbol{\mathrm{TC}},N,K)=\text{"Given a list of }\\{\mathrm{N}\\}\text{ labels, cluster them into }\\{\mathrm{K}\\}\text{ words"}+\boldsymbol{\mathrm{TC}}$
$\mathrm{P}\_{\text{step3}}(\boldsymbol{\mathrm{TC}})=\text{"Based on the image description, determine the most appropriate cluster"}+\boldsymbol{\mathrm{TC}}$

</div>

## Step 1: Extract Salient Features from The Image

Image에서 VLM을 통해 핵심 feature를 text 설명 형태로 추출한다.

> Step 1: Vision-language model (VLM) extracts salient features

<div style="overflow: auto;">
$$
\begin{aligned}
&\textbf{Input: }\text{Image Dataset }\mathcal{D}_{\mathrm{img}},\text{ Text Criteria }\boldsymbol{\mathrm{TC}},\text{ Descriptions }\mathcal{D}_{\mathrm{des}} \gets[\ ]\newline
&\textbf{Output: }\mathcal{D}_{\mathrm{des}}\newline
&1:\textbf{ for }\mathrm{img}\text{ in }\mathcal{D}_{\mathrm{img}}\textbf{ do}\newline
&2:\quad\mathcal{D}_{\mathrm{des}}.\text{append}(\mathrm{VLM}(\mathrm{img},\mathrm{P}_{\text{step1}}(\boldsymbol{\mathrm{TC}})))\quad//\text{ append image description to }\mathcal{D}_{\mathrm{des}}\newline
&3:\textbf{end for}
\end{aligned}
$$
</div>

사용자의 기준인 $\boldsymbol{\mathrm{TC}}$는 VLM이 집중해야하는 관련 feature를 결정한다.
예를 들어, 사용자는 아래와 같이 image 속 인물 또는 장면의 전반적 분위기를 원할 수 있다.

+ Criterion 1: Focus on the mood of the person in the center.
+ Criterion 2: Describe the general mood by inspecting the background.

## Step 2: Obtaining Cluster Names

LLM에서 두 개의 하위 단계를 통해 cluster의 이름을 검색한다.
2a 단계에서는 LLM은 image의 raw initial label을 출력한다.
Raw initial label의 수는 일반적으로 $K$보다 크므로 2b 단계에서 LLM은 raw initial label을 $K$ cluster의 적절한 이름으로 집계한다. (2a 단계와 2b 단계를 결합하고 $N$개의 image 설명에서 $K$개의 cluster 이름을 검색하도록 LLM에 요청하는 것은 LLM의 제한된 token 길이로 불가능)

> Step 2: Large Language Model (LLM) obtains $K$ cluster names

<div style="overflow: auto;">
$$
\begin{aligned}
&\textbf{Input: }\text{Descriptions }\mathcal{D}\_{\mathrm{des}},\text{ Text Criteria }\boldsymbol{\mathrm{TC}},\text{ Dataset size }N,\text{ Nubmer of clusters } \mathcal{L}\_{\mathrm{raw}}\gets[\ ]\newline
&\textbf{Output:}\text{ List of cluster names }\mathcal{C}\_{\mathrm{name}}\newline
&1:\textbf{ for }\mathrm{description}\text{ in }\mathcal{D}\_{\mathrm{des}}\textbf{ do}\newline
&2:\quad\mathcal{L}\_{\mathrm{raw}}.\text{append}(\mathrm{LLM}(\mathrm{description}+\mathrm{P}\_{\text{step2a}}(\boldsymbol{\mathrm{TC}})))\quad//\text{ append raw label to }\mathcal{L}\_{\mathrm{raw}}\newline
&3:\textbf{end for}\newline
&4:\mathcal{C}\_{\mathrm{name}}=\mathrm{LLM}(\mathcal{L}\_{\mathrm{raw}}+\mathrm{P}\_{\text{step2b}}(\boldsymbol{\mathrm{TC}},N,K))\quad//\text{ Step 2b can be further optimized}
\end{aligned}
$$
</div>

2b 단계의 가장 간단한 instance는 raw label의 전체 목록인 $\mathcal{L}\_{\mathrm{raw}}$를 직접 제공한다.
하지만 저자들은 $\mathcal{L}\_{\mathrm{raw}}$를 key와 발생하는 횟수를 가지는 dictionary로 변환하는게 더욱 효율적임을 발견했다.
동일한 raw label이 여러번 발생하는 경우 이 최적화를 통해 2b 단계의 LLM에 대한 입력 token 길이가 크게 줄어든다.
$\mathrm{P}_{\text{step2b}}(\boldsymbol{\mathrm{TC}},N,K)$의 prompt engineering을 통해 사용자는 cluster를 사용자 기준과 일치하도록 세분화 할 수 있다.
예를 들어, 아래와 같이 사용자는 추가 text prompt를 추가할 수 있다.

```
When categorizing the classes, consider the following criteria:
    1. Merge similar clusters. For example, [sparrow, eagle, falcon, owl, hawk] should be combined into 'birds of prey.'
    2. Clusters should be differentiated based on the animal’s habitat.
```

## Step 3: Clustering by Assigning Images

Image가 최종 $K$ cluster 중 하나에 할당된다.
Text criterion $\boldsymbol{\mathrm{TC}}$, 1단계의 text description, 2단계의 $K$ cluster 이름이 LLM에 제공된다.

> Step 3: Large Language Model (LLM) assigns clusters to images

<div style="overflow: auto;">
$$
\begin{aligned}
&\textbf{Input: }\text{Descriptions }\mathcal{D}_{\mathrm{des}},\text{ Text Criteria }\boldsymbol{\mathrm{TC}},\text{ List of cluster names }\mathcal{C}_{\mathrm{name}},\ \mathrm{RESULT}\gets[\ ]\newline
&\textbf{Output: }\mathrm{RESULT}\newline
&1:\textbf{ for }\mathrm{description}\text{ in }\mathcal{D}_{\mathrm{des}}\textbf{ do}\newline
&2:\quad\mathrm{RESULT}.\text{append}(\mathrm{LLM}(\mathrm{description}+\mathrm{P}_{\text{step3}}(\boldsymbol{\mathrm{TC}})))\quad//\text{ append assigned cluster}\newline
&3:\textbf{end for}
\end{aligned}
$$
</div>

## Iteratively Editing The Algorithm Through Text Prompt Engineering

> Main method: IC|TC

<div style="overflow: auto;">
$$
\begin{aligned}
&\textbf{Input: }\text{Dataset }\mathcal{D}_{\mathrm{img}},\text{ Text Criteria }\boldsymbol{\mathrm{TC}},\ \mathrm{ADJUST}\gets\mathrm{True}\newline
&1:\textbf{ while }\mathrm{ADJUST}\textbf{ do}\newline
&2:\quad\mathrm{RESULT}\gets\textbf{do Steps 1-3}\text{ conditioned on }\boldsymbol{\mathrm{TC}}\newline
&3:\quad\textbf{if}\text{ User determines }\mathrm{RESULT}\text{ satisfactory }\textbf{then}\newline
&4:\quad\quad\mathrm{ADJUST}\gets\mathrm{False}\newline
&5:\quad\textbf{else}\newline
&6:\quad\quad\boldsymbol{\mathrm{TC}}\gets\text{Update }\boldsymbol{\mathrm{TC}}\quad//\text{ user writes updated }\boldsymbol{\mathrm{TC}}\newline
&7:\quad\textbf{end if}\newline
&8:\textbf{end while}
\end{aligned}
$$
</div>

한 번 clustering을 수행한 후 cluster가 지정된 text 기준 $\boldsymbol{\mathrm{TC}}$와 충분히 일치하지 않거나 사용자가 염두에 둔 내용을 $\boldsymbol{\mathrm{TC}}$가 정확히 지정하지 않은 것으로 판명되면 사용자는 $\boldsymbol{\mathrm{TC}}$를 update할 수 있다.
이러한 반복 process는 사용자의 판단 하에 clustering 결과가 만족스러울 때까지 계속될 수 있다.

## Producing Cluster Labels

일반적으로 비지도 clustering 작업에는 출력 cluster의 label이나 설명을 생성하는 방법이 필요하지 않다.
그러나 IC|TC는 cluster를 설명하는 이름을 생성하고 이는 clustering 결과를 상대적으로 직접적이고 즉각적으로 해석할 수 있게 한다.

---

# Experiments

IC|PC는 foundation model, 특히 instruction tuning을 거친 VLM 및 LLM의 사용에 크게 의존한다.
저자들은 실험에 주로 VLM에 [LLaVa](https://arxiv.org/abs/2304.08485), LLM에 [GPT-4](https://arxiv.org/abs/2303.08774)를 사용했고 foundation model의 변화에 따른 성능 변화를 파악하기 위한 ablation study를 진행했다.

|Dataset|Criterion|SCAN|IC\|TC|
|:-:|:-:|:-:|:-:|
|Stanford 40 Action|`Action`|0.397|**0.774**|
|Stanford 40 Action|`Location`|0.359|**0.822**|
|Stanford 40 Action|`Mood`|0.250|**0.793**|
|PPMI|`Musical Instrument` ($K=7$)|0.632|**0.964**|
|PPMI|`Musical Instrument` ($K=2$)|0.850|**0.977**|
|PPMI|`Location` ($K=2$)|0.512|**0.914**|
|CIFAR-10-Gen|`Object`|**0.989**|0.987|

## Clustering with Varying Text Criteria

본 실험을 통해 text 기준 $\boldsymbol{\mathrm{TC}}$를 변경하면 단일 image dataset의 clustering 결과가 다양해짐을 확인할 수 있다.
저자들은 이 결과를 통해 IC|TC가 매우 유연하고 다양한 text 기준을 수용할 수 있다고 주장한다.
9,532장의 image와 독서, 전화 걸기, 거품 불기, violin 연주 등과 같은 40개 class 중 대상의 행동 (`Action`)을 설명하는 image label을 포함하는 [Stanford 40 Action Dataset](https://ieeexplore.ieee.org/document/6126386)을 사용했다.
그리고 저자들은 restaurant, store, sports facility 등 위치 (`Location`)를 설명하는 10개의 class를 포함하는 collection과 joyful, adventurous, relaxed, focused와 같이 장면의 분위기 (`Mood`)를 나타내는 4개의 class를 포함하는 collection을 추가로 정의했다. (`Location`과 `Mood`에 대한 ground truth는 없기 때문에 1,000장의 randomly sampled image들에 대해 labeling 수행)
3가지 개별 clustering 결과를 얻기 위해 3가지 text 기준 (`Action`, `Location`, `Mood`)를 활용한다.
저자들은 IC|TC를 3가지 label collection을 얼마나 정확하게 복구하는지에 따라 평가했고 정량적 비교를 위해 기존의 deep clustering 방법인 [SCAN](https://arxiv.org/abs/2005.12320)을 선택했다.

## Clustering with Varying Granularity

본 실험을 통해 IC|TC가 cluster 수인 $K$를 조정하여 clustering 결과의 세분성을 자동으로 제어할 수 있음을 보여준다.
저자들은 IC|TC가 출력한 cluster 설명이 해석하기 쉽고 다양한 $K$ 값에 대해 image가 cluster에 할당된다는 것을 발견했다.
이를 정량적으로 검증하기 위해 12개의 서로 다른 악기와 상호 작용하는 사람의 1,200개의 image를 포함하는 [People Playing Musical Instrument (PPMI) dataset](https://ieeexplore.ieee.org/document/5540018)를 사용했고 task의 크기와 난이도를 줄이기 위해 7개의 class에 대해 700장의 image를 선택했다.
Cluster의 수 $K=2$와 $K=7$에 대해 text 기준 `Musical Instrument`을 사용하여 실험을 진행한 결과 $K=7$을 사용하면 image는 violin, guitar와 같은 cluster로 군집화되며 실제 label에 대해 96.4%의 정확도를 달성하고, $K=2$를 사용하면 image는 금관악기와 현악기의 2개 cluster로 나뉘며 97.7%의 정확도를 달성한다.
저자들은 별도로 IC|TC에 7개의 악기를 금관악기와 현악기로 grouping하도록 구체적 지시를 하지 않았고 이러한 hierarchical grouping은 IC|TC에 의해 발견되었다.
추가적으로 PPMI dataset에 대해 text 기준 `Location`과 $K=2$을 사용한 clustering은 실내와 실외로 나뉜다.

## Comparison with Classical Clustering Methods

본 실험을 통해 IC|TC를 CIFAR-10, STL-10, CIFAR-100의 여러 기존 clustering algorithm과 비교한다.
3가지 dataset에는 각각 10, 10, 20개의 class와 10,000, 8,000, 10,000장의 image가 있다.
Dataset의 class 수와 동일한 cluster 수를 가지는 text 기준 `Object`를 사용했고 아래와 같이 결과적으로 모든 dataset에 대해 기존 clustering 방법보다 훨씬 뛰어난 성능을 달성했다.
기존 clustering 방법은 foundation model이나 pre-trained 가중치를 사용하지 않기 때문에 IC|TC와의 비교가 불공평하지만 foreground object type을 기반으로 image를 clustering하는 것이 목표일 때 IC|TC가 경쟁력있음을 보여준다.

|Method|ACC<br />(CIFAR-10)|NMI<br />(CIFAR-10)|ARI<br />(CIFAR-10)|ACC<br />(STL-10)|NMI<br />(STL-10)|ARI<br />(STL-10)|ACC<br />(CIFAR-100)|NMI<br />(CIFAR-100)|ARI<br />(CIFAR-100)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[IIC](https://arxiv.org/abs/1807.06653)|0.617|0.511|0.411|0.596|N/A|N/A|0.257|N/A|N/A|
|[SCAN](https://arxiv.org/abs/2005.12320)|0.883|0.797|0.772|0.809|0.698|0.646|0.507|0.468|0.301|
|[SPICE](https://arxiv.org/abs/2103.09382)|0.926|0.865|0.852|0.938|0.872|0.870|0.584|0.583|0.422|
|[RUC](https://arxiv.org/abs/2012.11150)|0.903|N/A|N/A|0.867|N/A|N/A|0.543|N/A|N/A|
|[TCL](https://arxiv.org/abs/2210.11680)|0.887|0.819|0.780|0.868|0.799|0.757|0.531|0.529|0.357|
|IC\|TC<br />([LLaVA](https://arxiv.org/abs/2304.08485) only)|0.647|0.455|0.442|0.774|0.587|0.589|0.097|0.022|0.014|
|IC\|TC<br />([LLaVA](https://arxiv.org/abs/2304.08485) + [Llama 2](https://arxiv.org/abs/2307.09288))|0.884|0.789|0.759|0.974|0.939|0.944|0.526|0.554|0.374|
|IC\|TC<br />([BLIP-2](https://arxiv.org/abs/2301.12597) + [GPT-4](https://arxiv.org/abs/2303.08774))|**0.975**|**0.941**|**0.947**|**0.993**|**0.982**|**0.985**|0.584|**0.690**|**0.429**|
|IC\|TC<br />([LLaVA](https://arxiv.org/abs/2304.08485) + [GPT-4](https://arxiv.org/abs/2303.08774))|0.910|0.823|0.815|0.986|0.966|0.970|**0.589**|0.642|0.422|

## Fair Clustering Through Text Criterion Refinement

기존 clustering 방법은 편향된 결과를 보이는 경우가 존재하며 이러한 현상을 완화가이 위한 조치들이 연구되었다.
Foundation model은 train data에서 편향을 학습하는 것으로 알려져 있기 때문에 IC|TC는 이러한 편향을 clustering 결과에 전파할 위험이 있다.
본 실험에서는 text 기준에 "Do not consider gender"라는 prompt를 추가하기만 하면 clustering 결과의 편향을 효과적으로 완화할 수 있음을 보여준다.
[FACET](https://arxiv.org/abs/2309.00035)은 52개의 직업 class를 포함하여 여러 속성이 표시된 32,000장의 다양한 image로 구성되어있는 AI 및 machine learning vision model의 견고성과 algorithm의 공정성을 평가하기 위한 benchmark dataset다.
저자들은 본 실험을 진행하기 위해 craftsman, laborer, dancer, gardener 직업군에서 남성과 여성 각각 20장의 image를 sampling하여 총 160장의 image dataset을 구성했다.
Text 기준 `Occupation`을 사용할 때 IC|TC는 gender bias를 나타냈고 이러한 편견을 완화하기 위해 저자들은 IC|TC에 성별을 고려하지 않고 activity에 집중하도록 지시하는 간단한 부정적 prompt를 도입했다.
Clustering을 반복한 결과 craftsman cluster와 laborer cluster의 성비 격차는 각각 27.2%에서 4.4%, 11.6%에서 3.2%로 개선되었다.
또한 dancer cluster와 gardner cluster도 각각 2.8%에서 2.6%, 10.6%에서 9.0%로 격차가 소폭 감소했다.

## Further Analyses

### Ablation Studies of LLMs and VLMs

VLM만으로 충분한지 그리고 LLM이 실제로 IC|TC에서 중요한 역할을 하는지 평가하기 위해 ablation study를 수행한다.

+ LLaVA only vs. LLaVA + LLM (Llama 2, GPT-4): LLM을 사용하지 않으면 (LLaVA only) 성능이 매우 좋지 않고 LLM을 사용하면 성능이 크게 향상
+ LLaVA + Llama 2 vs. LLaVA + GPT-4: LLM의 크기에 따른 성능 변화는 상대적으로 적음
+ BLIP-2 + GPT-4 vs. LLaVA + GPT-4: Blip-2와 LLaVA는 text 기준과 관련된 정보를 추출할 수 있기 때문에 높은 성능 달성 (Image captioning model인 [ClipCap](https://arxiv.org/abs/2111.09734)은 text conditioning을 수행할 수 없어 성능 저하 발생)

이를 통해 저자들은 IC|TC에서 LLM이 중요한 역할을 (VLM 단독으로는 충분하지 않음) 하지만 LLM의 크기는 상대적으로 중요하지 않다고 주장한다.

### Data Contamination

Foundation model을 사용하여 연구를 평가할 때 data 오염 가능성은 중요하다.
정확도를 측정하기 위한 dataset인 CIFAR-10, STL-10, CIFAR-100, Stanford 40 Action는 LLaVA의 학습 시 사용되었을 수 있다.
따라서 저자들은 정확도 측정의 타당성이 떨어진다 판단하여 [Stable Diffusion XL](https://arxiv.org/abs/2112.10752)과 CIFAR-10 label을 사용하여 1,000장의 CIFAR-10 유사 image를 생성하고 해당 dataset을 CIFAR-10-Gen으로 명명했다.
저자들은 해당 dataset에 대해 IC|TC는 98.7%의 정확도를 달성했으며 CIFAR-10-Gen dataset의 정확도가 CIFAR-10 dataset의 정확도보다 나쁘지 않다는 사실을 통해 IC|TC의 높은 성능이 data 오염으로 인한 것이 아닐 가능성이 높다고 주장한다.
