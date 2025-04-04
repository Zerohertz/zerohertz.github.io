---
title: Statistics
date: 2020-01-07 14:03:55
categories:
- Etc.
tags:
- R
- Statistics
- B.S. Course Work
---
# 통계적 추론

## 가설 검정
> 표본을 통해 모집단의 특성(모수`mu`)를 알기위해

+ 대문자 `X` - rv(확률변수)
+ 소문자 `x` - 표본(하나의 값)

## rv - 통계량 - 표본분포(통계량의 확률분포)
<!-- more -->
***
# 확률변수

## 표본공간의 각 사건을 수치로 대응시켜주는 함수(rv - random variable, `X`) - function
+ 정수 - 이산형
+ 실수 - 연속형

## `ex`) 동전 2개
+ `HH`, `HT`, `TH`, `TT`(원소, 사건) - 표본공간(전체집합)
+ 앞면의 수 - 함수
    + X(`HH`) = 2
    + X(`HT`) = 1
    + X(`TH`) = 1
    + X(`TT`) = 0
    + X(`정의역`)=`치역`
+ `X` - 값을 여러개 취함
+ 대문자는 확률변수, 소문자는 값

## `ex2`) n=1
+ 시행 1번
+ `X`: 앞면의 수
+ p(X=x)=(p^x)*((1-p)^(1-x))

|   `X`    |              0               |              1               |
| :------: | :--------------------------: | :--------------------------: |
| `p(X=x)` | $\displaystyle{\frac{1}{2}}$ | $\displaystyle{\frac{1}{2}}$ |

## 기댓값, 평균
+ 기댓값 - 확률변수의 평균
+ 평균 - Data의 산술평균

***
# 확률분포

## 확률변수의 분포
+ 표
+ Graph
+ Function

## 확률 function
+ 확률 질량 함수(이산) - `p(X=x)`
+ 확률 밀도 함수(연속) - `f(x)`

***
# 베르누이 시행

+ 결과 : 단 두개 - 성공 / 실패
+ `p(성공)`=p, `p(실패)`=1-p=q
+ 각 시행은 독립시행(전사건이 후사건에 영향 X)

***
# 이산형 분포

|   분포   | ?의 횟수 | 베르누이 시행 |           기호(X~)            |               X                |                         p(X=x)                          |     R      |
| :------: | :------: | :-----------: | :---------------------------: | :----------------------------: | :-----------------------------------------------------: | :--------: |
| 베르누이 |   성공   |       O       |             Be(p)             |  1번의 베르누이 시행 성공횟수  |          $\displaystyle{p^x\times(1-p)^{1-x}}$          | `binom()`  |
|   이항   |   성공   |       O       |            B(n,p)             |  n번의 베르누이 시행 성공횟수  |     ${}_n \mathrm{C}_x\times p^x\times(1-p)^{n-x}$      | `binom()`  |
|   기하   |   시행   |       O       |             G(p)              |    처음 성공까지의 시행횟수    |            $\displaystyle{q^{x-1}\times p}$             |  `geom()`  |
|  음이항  |   시행   |       O       |            NB(k,p)            |    k번 성공까지의 시행횟수     | $(x-1)C(k-1) \times p^{k-1} \times (1-p)^{x-k}\times p$ | `nbinom()` |
|  포아송  |   성공   |       O       | $\displaystyle{P_0(\lambda)}$ | 단위(시간, 면적, ...) 성공횟수 | $\displaystyle{e^{-\lambda} \times \lambda^x\over x!}$  |  `pois()`  |
|  초기하  |   성공   |       X       |           HG(N,n,D)           |            성공횟수            |       ${}_DC_x \times (N-D)C(n-x) \over {}_N C_n$       | `hyper()`  |

## 이항분포
+ E(X)=np - 평균(성공횟수)
+ V(X)=npq

## 기하분포
+ 무한하게 시행시 무조건 성공(`p/(1-q)=1` by 무한등비급수)
+ E(X)=1/p - 평균(시행횟수)
+ V(X)=q/(p^2)

## 음이항분포
+ 이항분포의 반대(`시행횟수` <-> `성공횟수`)
+ E(X)=k/p - 평균(시행횟수)
+ V(X)=k*q/(p^2)

## 포아송분포
+ `n->inf`, `p->0` 근사
+ `ex`) p=0.001, n=3000
    + p(X=5)=3000C5((0.001)^5)*(0.999)^2995
    + p(X=x)=(e^(-lambda)*lambda^x)/(x!)
        + lambda=np
        + p=(lambda)/n
        + 이항분포에 근사하여 대입 후 증명
+ `lambda`- 평균성공횟수
+ E(X)=lambda

## 초기하분포
+ 독립시행 X - 베르누이 시행을 따르지 않음
+ 단순 확률 구하기와 Similar
+ 변수
    + N - 전체집단 개수
    + n - 추출대상 개수
    + D - 성공집단 개수

***
# 연속형 분포

## 지수분포
+ `ex`) 처음 죽을때까지의 시간

***
# R

## 표

|          mean          |                                R                                |
| :--------------------: | :-------------------------------------------------------------: |
|        이항분포        |                             binom()                             |
|        기하분포        |                             geom()                              |
|       음이항분포       |                            nbinom()                             |
|       포아송분포       |                             pois()                              |
|       초기하분포       |                             hyper()                             |
|    표본추출 - 이항     |               `rbinom(표본수,시행횟수,성공확률)`                |
|    `p(X=n)` - 이항     |              `dbinom(성공횟수,시행횟수,성공확률)`               |
|    `p(X<=n)` - 이항    |              `pbinom(성공횟수,시행횟수,성공확률)`               |
|    `p(X=n)` - 기하     |                   `dgeom(실패횟수,성공확률)`                    |
|   `p(X=n)` - 음이항    |              `dnbinom(실패횟수,성공횟수,성공확률)`              |
|   `p(X=n)` - 포아송    |                    `dpois(성공횟수,lambda)`                     |
|   `p(X<=n)`- 포아송    |                    `ppois(성공횟수,lambda)`                     |
|    `p(X=n)`- 초기하    | `dhyper(성공횟수(x),성공표본수(D),다른표본수(N-D),추출개수(n))` |
|         t 분포         |                               t()                               |
|         F 분포         |                               f()                               |
|        정규분포        |                             norm()                              |
|    정규분포 함숫값     |                            `dnorm()`                            |
|     정규분포 누적      |                            `pnorm()`                            |
| 정규분포 x,z값(분위수) |                       `qnorm(누적확률값)`                       |
|        지수분포        |                             `exp()`                             |

+ `r` - 추출
+ `d` - 확률
+ `p` - 누적 확률
+ `q` - 분위수

## R source

~~~R
> ### 이산형 ###
> dbinom(1,3,0.5)
[1] 0.375
> pbinom(2,3,0.5)
[1] 0.875
> dbinom(0,3,0.5)+dbinom(1,3,0.5)+dbinom(2,3,0.5)
[1] 0.875
> dgeom(2,0.4)
[1] 0.144
> dnbinom(1,2,0.4)
[1] 0.192
> dpois(2,3)
[1] 0.2240418
> dpois(0,3)+dpois(1,3)+dpois(2,3)
[1] 0.4231901
> ppois(2,3)
[1] 0.4231901
> dhyper(2,3,2,3) # B=3, W=2 3개 추출, B=2, W=1
[1] 0.6
> ### 연속형 ###
> x=seq(-3,3,length=100)
> x
  [1] -3.00000000 -2.93939394 -2.87878788 -2.81818182 -2.75757576 -2.69696970 -2.63636364 -2.57575758
...
 [97]  2.81818182  2.87878788  2.93939394  3.00000000
> plot(x,dnorm(x)) # 표준정규분포 확률밀도함수
> plot(x,dnorm(x),type='l') # type = line
> x=rnorm(100000000)# 랜덤생성
> mean(x);sd(x)
[1] -7.529876e-05
[1] 1.000055
> y=rnorm(100000000,2000,10)
> mean(y);sd(y)
[1] 2000
[1] 10.00069
> hist(x)
> pnorm(0)
[1] 0.5
> pnorm(0)-pnorm(-1.96) # p(-1.96<X<0)
[1] 0.4750021
> pnorm(110,100,5)-pnorm(90,100,5) # p(90<X<110)
[1] 0.9544997
> qnorm(0.975) ############
[1] 1.959964
> qnorm(0.5,175,10)
[1] 175
~~~

> plot(x,dnorm(x))
![plot-dnorm](/images/r-statistics/plot-dnorm.png)

> plot(x,dnorm(x),type='l')
![plot-dnorm-l](/images/r-statistics/plot-dnorm-l.png)

> hist(x)
![hist](/images/r-statistics/hist.png)

***
# 표본분포

## Xbar - 통계량(Statistic) - r.v

+ 대문자
+ 값 여러개 대응
+ 분포를 가지고 있음
+ `xbar` - 표본의 평균이므로 소문자
+ `Xbar` - 표본평균의 평균이므로 대문자
+ 불편성(E(X)=E(`Xbar`) - 추정량의 평균과 실제 모수값과 일치)을 만족하기때문에 `Xbar` 사용

### `ex`) 모집단 - `1,2,3`, 추출 - `2`

> 통계량의 확률분포는 표본분포(`Xbar`의 확률분포는 표본분포)

|      `xbar`      |   1   |  1.5  |   2   |  2.5  |   3   |
| :--------------: | :---: | :---: | :---: | :---: | :---: |
| p(`Xbar`=`xbar`) |  1/9  |  2/9  |  3/9  |  2/9  |  1/9  |

+ `Xbar`=1/2*(Xsub1+Xsub2) - Xsub1, Xsub2는 모집단의 분포를 따름
+ E(`Xbar`)=E(1/2*(Xsub1+Xsub2))=1/2*(E(Xsub1)+E(Xsub2))

## 정규모집단(정규분포)

> X ~ N(`mu`,`sigma`^2)

+ 모집단 분포 - X ~ N(`mu`,`sigma`^2)
+ 모집단 분포 - Xsub1, Xsub2, ..., Xsubn ~ N(`mu`,`sigma`^2)
+ E(`Xbar`)=1/n(`mu`+...+`mu`)=`mu`
+ V(`Xbar`)=1/(n^2)*(`sigma`^2+...+`sigma`^2)=`sigma`^2/n
+ `Xbar` ~ N(`mu`,`sigma`^2/n) - 모집단이 정규분포기 때문
+ `X`, `mu`, `sigma` 추정

## 모집단 - 임의의 분포

> E(X)=`mu`, V(X)=`sigma`^2

### 중심극한정리(CLT)

+ n>=30
+ `Xbar` ~ N(`mu`,`sigma`^2/n)
+ 모집단이 정규분포를 따르지 않아도 표본집단이 30 이상일 경우 `Xbar`는 정규분포
+ `ex`) `mu1`-`mu2` 평균의 차이 분석 - `Xbar1`-`Xbar2` ~ N 이용


***
# 가설검정

## 1. 가설설정

![hypothesis](/images/r-statistics/hypothesis.png)

+ `Hsub0`를 채택, 기각(무조건 `Hsub0` 중심으로 서술)

### 귀무가설(Hsub0)

> 연구자가 기각하길 바라면서 세운 가설

+ 검정대상이 되는 가설
+ `Hsub0`: ~이다, ~차이가 없다, ~영향력이 없다,`=`

### 대립가설(Hsub1)

> 연구자가 증명하고자 하는 가설

+ 귀무가설이 거짓이면 대립가설 참
+ `Hsub1`: ~아니다, ~차이가 있다, ~영향력이 있다, `=/=`(양측검정), `>`, `<`(단측검정)
+ 양측검정 - 사전정보가 없어서 양쪽 확인
+ 단측검정 - 사전정보가 있어서 한쪽만 확인


## 2. 유의수준

> 가설검정시 허용되는 오류

+ `alpha=0.05`, 0.01, 0.001
+ $\alpha=0.05$이면 5%의 오류 허용

## 3. 검정통계량 계산

+ $\mu_0$를 설정하여 정규분포표를 이용해 Normalize

## 4. 유의수준과 유의확률 비교 / 검정통계량값과 기각역 비교(학부 X) 

> 유의수준 $\geq$ 유의확률 - `Hsub0` 기각

### 유의수준

> $$\alpha$$

### 유의확률(p-value)

> $H_0$를 기각하게 하는 최소의 유의수준

+ 단측검정 - 검정통계량값의 적은쪽 꼬리확률
+ 양측검정 - 검정통계량값의 적은쪽 꼬리확률X2


## 5. 결론 및 해석

+ 위의 flow대로 번호써서 진행
+ 검정통계 공식, 값

***
# 분석기법

+ 일표본 T-검정 - $H_0$ : $\mu$ = float
    + 수치자료 1개
+ 독립표본 T-검정(두 집단의 평균비교) - $H_0$ : $\mu_1$ - $\mu_2$ = 0
    + 집단변수(범주형, factor, 집단 2개), 수치자료 1개
+ 대응표본 T-검정(변화량 비교) - $H_0$ : $\mu_{before}$ - $\mu_{after}$ = 0
    + 수치자료 2개
+ 일원배치 분산분석(세 집단 이상의 평균비교) - $H_0$ : $\mu_1$ = $\mu_2$ = ... = $\mu_k$
    + 집단변수(범주형, factor, 집단 3개 이상), 수치자료 1개
+ 회귀분석
    + 독립변수 : 종속변수에 영향을 주는 변수(수치자료 n개)
    + 종속변수 : 독립변수에 영향을 받는 변수(수치자료 1개)
+ 교차분석
    + 범주형자료 2개
+ 상관분석
    + 수치자료 2개