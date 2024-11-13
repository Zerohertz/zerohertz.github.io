---
title: 'Lecture Review: 실전 AI 서비스 기획 (Jump into the AI World - AI Product Lifecycle)'
date: 2024-01-19 08:45:19
categories:
- 4. MLOps
tags:
- Lecture Review
---
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297024102-028740af-f0a1-401b-8629-52069c9831da.png" width=600>

# Introduction

Upstage에서 진행하는 "[Jump into the AI World - AI Production Lifecycle](https://bit.ly/up-amb-jump-into-the-ai-world)" 강의의 홍보대사로 선발되어 해당 강의를 수강할 수 있는 감사한 기회를 얻게되었습니다.

<!-- More -->

강의를 살펴보기 전에 Upstage에 대해 간단히 알아봅시다!

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297023908-e2176ebc-6122-4015-85c3-763b4bd08a9b.png" width=300>

Upstage는 2020년에 전 Naver CLOVA AI head이자 홍콩과학기술대학교 교수이신 [Sung Kim](https://github.com/hunkim)님을 필두로 이활석 CTO 님과 박은정 CSO 님이 함께 뜻을 함께하여 창립되었습니다.
주요한 product 및 buisness model은 [Document AI](https://www.upstage.ai/feed/product/document-ai-ocr-for-llm) ([전 AI pack (AI OCR, RecSys)](https://youtu.be/o1A9qVTc_vs?feature=shared)), 자체 개발 LLM model [Solar (10.7B)](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0), [Edu stage](https://www.content.upstage.ai/edustage) (AI 교육) 등이 존재합니다.
ChatGPT의 등장 이후 시작된 LLM 시대에서 [Llama 2 (70B)를 fine tuning](https://www.upstage.ai/feed/company/huggingface-llm-no1-interview)하고 [자체 개발 LLM model Solar](https://www.upstage.ai/feed/insight/top-open-source-llms-2024)를 개발하여 hugging face에서 1위를 달성해 기술력을 드러내고 있습니다.

Upstage에서 CTO (최고 기술 경영자)인 이활석님이 Udemy에서 진행한 강의의 제목은 "Jump into the AI World - AI Production Lifecycle"이고 포함되는 내용들은 아래와 같습니다.

+ ‘AI 기술을 제대로 활용할 수 있는가’에 대한 제대로 된 고민을 통해 AI 기술과 생태계, 비즈니스에 대한 지식을 학습할 수 있습니다.
+ AI 제품 기획을 위한 A to Z 과정의 이해도를 높여, AI 비즈니스 Use case 발굴 및 제안할 수 있는 역량을 강화할 수 있습니다.
+ AI 기술이 서비스와 비즈니스에 활용되는 사례를 이해하고, AI/ML기반의 서비스 기획 역량을 향상시킬 수 있습니다.
+ AI 서비스 개발부터 배포 후 관리까지 과정별 필요한 역할을 이해함으로써 실무에 바로 적용할 수 있는 핵심 기술을 익힐 수 있습니다.
+ AI 비즈니스에 적합한 서비스 기획과 시뮬레이션 경험을 통해 실무 감각을 배양할 수 있습니다.

해당 강의의 대상은 아래와 같습니다.

+ AI 분야에서 커리어를 시작하고 싶거나 AI 생태계가 어떻게 돌아가는지 궁금하신 분
+ AI 기술을 활용한 제품과 서비스 아이디어를 내기 위해 필요한 기초 지식을 얻고 싶으신 분
+ 제품과 서비스에 AI 기술을 도입하고 싶은 실무 담당자 (기획자, 마케터, 사업개발 담당자, 엔지니어 등)
+ ‘AI 원리가 무엇인지', ‘AI 기술을 우리 회사 혹은 서비스에 어떻게 적용할 수 있을지’, ‘AI 개발자들과 소통을 잘하기 위해 그들은 어떻게 일하고 고민하는지’를 알고 싶으신 분

---

# Review

## Orientations

강의는 아래와 같은 특징들을 설명하며 시작합니다.

> "우리 강의는 인공지능 기술 및 비즈니스 세계에 첫 걸음을 내딛고 싶으신 모든 분들을 위한 강의입니다."

+ 강의 목표
  + "AI 기술을 제대로 활용할 수 있는가"에 대한 고민에서 그치지 않고 AI 기술 및 생태계, 비즈니스에 대한 지식 학습
  + AI 제품 및 서비스 기획을 위한 전체 과정의 이해도 향상을 통한 AI buisness use case 발굴 및 제안 역량 강화
+ 추가 정보
  + 강의 시간: 8시간
  + 수강생 사전 지식: 선수과정이 요구되지 않음

<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297953208-ebd773c6-e399-4ba0-94c1-66d1a392506e.png" width=300>

강의의 목표는 위의 quiz와 같이 "어떤 layer를 사용해서 SOTA를 달성하는지"와 같은 researcher 관점이 아닌 "AI product를 기획 및 개발"을 이해하는 것입니다.


## AI 기술의 흐름과 원리

`Session 1`은 AI의 기본 원리와 AI product의 life cycle을 이해하는 것을 목표로 합니다.

번역기, 이미지 생성 및 ChatGPT와 같은 대중들에게 잘 알려진 AI product부터 AI 관련 개발자가 아니라면 알기 어려운 use cases까지 다양한 예제를 설명하며 적재적소의 AI product가 어떤 효과를 불러올 수 있는지 알려줍니다.
AI product가 어떻게 구성되고 개발되는지 알아보기 전 software program의 이해를 높히기 위해 아래와 같이 간단한 program부터 실생활에서 사용하는 예제들을 설명하는데 단순한 software program이 어떻게 AI product로 확장되는지 간략한 기술적 부분을 함께 설명합니다.

![upstage-udemy-2](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297953215-1f603186-6584-4dbc-97f2-f5db711993c6.png)

그리고 이러한 software program의 SW 1.0, SW 2.0, SW 3.0 방식에 대해 아래와 같이 정의가 무엇인지, 어떤 단계로 진행하는지 비교 분석합니다.
이러한 방법론의 차이를 빵 제조에 비유하여 어떤 차이점이 중요한지 상세하면서 이해하기 쉽게 강조합니다.

![upstage-udemy-3](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297953238-441b58dc-90ef-4def-9dd6-f78188cf1575.png)

SW 1.0, 2.0, 3.0 방법의 차이를 알았으니 AI 기본 원리를 알아보기 위해 SW 2.0 방법인 deep learning의 학습 방법을 설명하는데 수학적인 부분이 많지 않지만 아주 핵심적인 방법론을 명쾌하게 정의합니다.
예를 들어, pretraining과 fine tuning의 차이점과 필요성을 구체적인 예시와 함께 설명합니다.
또한 SW 3.0 방법의 초거대 model과 zero-shot, few-shot에 대해 이것이 왜 가능한지 기술적인 부분과 함께 SW 2.0 방법의 deep learning과 어떤 차이가 존재하는지 알려줍니다.

마지막으로 구체적인 sample case를 통해 AI prodcut를 개발하는데 있어 어떤 단계로 진행해야하는지 전체적인 운영 관점에서 설명합니다.

## 제품 개발 A to Z

`Session 2`는 AI 제품 개발에 대한 기본적 개념과 process를 이해하는 것을 목표로 합니다.

첫 번째로 제품 기획을 언제 어떻게 진행해야하는지 알려줍니다.
이 부분에서 AI가 만능은 아니기 때문에 왜 AI를 적용하고자 하는지에 대한 검증의 필요성을 언급하며 개발팀과의 의사소통을 강조합니다.
Product에 AI를 적용하기 전 해당 product의 아래와 같은 특성과 상황에 따라 적용 여부를 결정할 수 있습니다.

+ Heuristics, SW 1.0 (AI가 적합하지 않은 상황)
  + 출력에 대한 설명과 error의 예측 요구
  + 제한된 정보 제공
  + 기술의 중요성 < 시장 출시 시기의 중요성
  + 새로운 기술을 원하지 않는 경우
+ AI, SW 2.0 (AI가 적합한 상황)
  + 자동화를 통한 효율 개선
  + 미래 사건 예측
  + 맞춤형 추천 및 개인화
  + 비정형 data 분석 및 처리

위의 조건들 중 AI가 적합한 상황들이 직접적으로 이해하기 어려울 수 있기에 현재 산업에서 어떤 적용들이 왜 적합했는지 그리고 어떤 효과가 있었는지 설명합니다.
상세하고 구체적인 다양한 예제를 통해 실전에서 수강자들이 실제 product 기획 시 어떤 조건들을 살펴보고 판단해야하는지 알려줍니다.

만약 AI를 product에 적용하기로 했다면 개발자에게 요구사항을 전달해야하는데 어떤 방법으로 기획을 진행하는지 그리고 체계적 기획이 왜 필요한지 알려줍니다.
AI product 기획 및 개발을 위한 서비스 기획 / 사업 담당, AI 기술팀 manager, 개발자로 나누어진 다양한 communication 상황의 예제를 제시하며 효율적인 communication이 이뤄지지 않는 이유와 해결책을 상세히 설명합니다.
AI product 기획 시 효율적인 communication을 위해 고려해야할 사항들을 명확히 분류하고 각 직무에 따라 어떤 관점을 가지고 있는지 설명합니다.
특히 해결책을 설명할 때 이활석 CTO 님의 다양한 경험을 통한 insight를 볼 수 있습니다.
예를 들어, AI model의 개발을 위한 data를 준비할 때 data 수집과 labeling guide를 준비해야하고 어떤 관점에서 어떻게 준비해야하는지 근거와 함께 설명합니다.
그리고 각 고려야할 사항에 대해 아주 세부적으로 어떤 점들을 고민해야하고 어떠한 선택들이 존재하는지 정리합니다.

세부적으로 data의 품질을 강조하며 아래와 같이 해당 업무가 왜 어려운지 그리고 왜 중요한지 구체적인 예제와 함께 설명합니다.

![upstage-udemy-4](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297953241-28679034-7ff5-480e-9b91-f8e45b9a5703.png)

이러한 점들을 고려하여 어떻게 data를 준비하는지 각 단계인 data design, 원천 data 조사 및 수집, data scheme 정의 및 설계, data 준비, model training 및 data refine, 배포에 대해 상세히 설명합니다.
또한 AI product 관점에서 data의 윤리에 대해 알려줍니다.

Data를 모두 준비했다면 model 개발을 진행하는데, 그 전 미리 정의해야하는 부분들인 test 방법, 정량평가 metric가 존재합니다.
해당 부분들에 대해 실무적으로 어떤 점들을 유의해야하고 어떤 차이가 있는지 설명합니다.

Model 개발을 완료하면 product에 배포하는데, serving system을 어떻게 구성하고 고려해야하는 것들은 어떤게 있는지 아래와 같은 항목들을 정리합니다.

+ Cloud AI vs. Edge AI
+ Offline learning vs. Online learning
+ Batch inference vs. Online inference

모든 개발이 완료되어 product를 배포했을 때 아래와 같은 이유로 품질이 저조할 수 있음을 알려줍니다.

+ Model accuracy: 시간에 따라 model 성능 감소
+ Data drift: Model이 학습 시점과 적용 시점의 data의 특성 상이
+ Concept drift: 예측하고자 하는 대상의 속성, 관계 또는 pattern이 시간에 따라 변화
+ model drift: 학습 data에 없는 새로운 입력

각 원인에 대해 상세한 예제와 함께 설명 후 지속적인 monitoring의 중요성과 각 원인에 따른 monitoring 방법을 설명합니다.

지속가능한 AI product를 만들기 위해 지속적으로 올바른 mental model을 형성할 수 있도록 개발해야합니다. (Mental model: 특정 기술 혹은 service가 어떻게 작동할 것이라고 믿는 사고 과정)
올바른 mental model 형성을 위해 준비할 수 있는 부분인 적절한 설명, 오류 대응, feedback 수집 등 다양한 방법을 설명합니다.

## AI 서비스 개발 생태계

`Session 3`는 AI 생태계와 AI product가 나오기까지의 직무 간 역할을 이해하는 것을 목표로 합니다.

급변하는 AI 기술 및 시장의 생태계와 AI product를 개발하기 위한 product owner, data managing operation manager, AI research scientist, machine learning engineer, legal, QA, infra, security, MLOps software engineer, DevOps software engineer, data engineer, software engineer 등 다양한 직무를 설명합니다.
아래와 같이 한 사람이 덜 깊지만 다양한 직무의 일을 하는 경우, 한 사람이 한 직무의 일을 매우 깊게 하는 경우 등 engineering team의 세부적인 직무를 회사 규모에 따른 예시를 통해 상세하게 알려줍니다.

![upstage-udemy-5](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/297964938-f48fd3a8-19dd-4625-ae3a-b76390f623ea.png)

## Wrap Up

최종적으로 모든 강의의 매우 중요한 부분을 요약하며 마무리됩니다.
AI product를 기획, 개발, 유지보수하기 위해 AI product lifecycle (AI 제품 생애주기)를 거치는 과정에서 고려해야할 수많은 부분이 존재하고, 이 부분들을 이활석 CTO 님의 insight로 일목요연하게 이해할 수 있습니다.

---

# Conclusion

Machine learning research engineer로 1년을 겨우 채워가는 신입으로써 일하며 보고 배웠던 점들도 많았지만, CTO 혹은 제품을 기획하고 운영하는 입장에서 바라보는 AI를 배우며 시야가 넓어짐을 느꼈습니다.
회사 내에서 협업을 하거나 스스로 어떤 부분을 개선할 때 더욱 넓은 부분에서 다양한 것들을 고려하고 확인할 수 있는 지식이 쌓인 것이 만족스럽습니다.
또한 최근 개발자로의 덕목 중 communication 능력이 중요하다고 생각되는데, 강의를 통해 AI product를 개발하는 다양한 인력의 R&R을 이해하고 어떻게 소통하는 것이 좋은지 알 수 있었습니다.

개발자라면 AI product의 필수적인 주기를 이해하여 능동적인 생각을 할 수 있는 기폭제가 될 것이고, 비개발자라면 AI의 전반적인 이해를 높히고 engineering team과의 효율적인 소통이 가능할 것이라 생각합니다.
Upstage에서 진행하는 "[Jump into the AI World - AI Production Lifecycle](https://bit.ly/up-amb-jump-into-the-ai-world)"를 꼭 수강해보는 것을 추천드립니다.

> `#Jump_into_the_AI_World`, `#AI_Production_Lifecycle`, `#Upstage`, `#업스테이지`, `#AI교육`, `#PM`, `#서비스기획`, `#인공지능입문`

> 본 글은 [Upstage](https://www.upstage.ai/)에서 진행한 "Jump into the AI World 강의 홍보대사 (Ambassador)" 활동을 통해 작성된 글입니다.

---

> 추가적으로 Upstage 측에서 2024년 2월 1일 오전 11시 ~ 3월 3일 오전 11시까지 유효한 [20% 할인 쿠폰이 적용된 강의의 링크](https://www.udemy.com/course/upstage-jump-into-the-ai-world/?couponCode=UDEMY_AMB20)를 제공해주어 공유드립니다!
