---
title: R Final-term
date: 2020-01-16 12:36:15
categories:
- Etc.
tags:
- R
- Statistics
- B.S. Course Work
---
> 결측값 X

# read.table

~~~R
setwd("/Users/zerohertz")
text=read.table('Data.txt',header=T)
~~~
***
# Data 취사 선택

> indexing - []
<!-- More -->
~~~R
name[which(조건식),c('name1','name2',...)]
# which는 생략 가능
~~~

~~~R
> Cars93[which(MPG.city>30),c('Model','Origin')]
     Model  Origin
31 Festiva     USA
39   Metro non-USA
42   Civic non-USA
73  LeMans     USA
80   Justy non-USA
83   Swift non-USA
84  Tercel non-USA
> Cars93[MPG.city>30,c('Model','Origin')]
     Model  Origin
31 Festiva     USA
39   Metro non-USA
42   Civic non-USA
73  LeMans     USA
80   Justy non-USA
83   Swift non-USA
84  Tercel non-USA
> Cars93[which(Cylinders==4&Manufacturer=='Hyundai'),c('Model','Min.Price','Max.Price')]
     Model Min.Price Max.Price
44   Excel       6.8       9.2
45 Elantra       9.0      11.0
46  Scoupe       9.1      11.0
47  Sonata      12.4      15.3
> Cars93[Cylinders==4&Manufacturer=='Hyundai',c('Model','Min.Price','Max.Price')]
     Model Min.Price Max.Price
44   Excel       6.8       9.2
45 Elantra       9.0      11.0
46  Scoupe       9.1      11.0
47  Sonata      12.4      15.3
~~~

> subset(select=, subset=)

~~~R
subset(name,select=c(name1,name2,...),subset=(조건식))
# parameter의 select or subset은 하나만 생략 가능
~~~

~~~R
> subset(Cars93,select=Model,subset=(MPG.city>30))
     Model
31 Festiva
39   Metro
42   Civic
73  LeMans
80   Justy
83   Swift
84  Tercel
> subset(Cars93,select=Model,MPG.city>30)
     Model
31 Festiva
39   Metro
42   Civic
73  LeMans
80   Justy
83   Swift
84  Tercel
> subset(Cars93,Model,subset=MPG.city>30)
     Model
31 Festiva
39   Metro
42   Civic
73  LeMans
80   Justy
83   Swift
84  Tercel
~~~
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
# R

## 표

|          mean          |                                R                                |
| :--------------------: | :-------------------------------------------------------------: |
|        이항분포        |                             binom()                             |
|        기하분포        |                             geom()                              |
|       음이항분포       |                            nbinom()                             |
|       포아송분포       |                             pois()                              |
|       초기하분포       |                             hyper()                             |
|    표본추출 - 이항     |                   `rbinom(표본수,평균,분산)`                    |
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
***
# 범주자료

~~~R
> summary(Cars93$AirBags)
Driver & Passenger        Driver only               None 
                16                 43                 34 
> table(Origin,AirBags)
         AirBags
Origin    Driver & Passenger Driver only None
  USA                      9          23   16
  non-USA                  7          20   18
> t2=with(Cars93,table(Origin,Type));t2
         Type
Origin    Compact Large Midsize Small Sporty Van
  USA           7    11      10     7      8   5
  non-USA       9     0      12    14      6   4
> t3=xtabs(~Origin+Type,Cars93);t3
         Type
Origin    Compact Large Midsize Small Sporty Van
  USA           7    11      10     7      8   5
  non-USA       9     0      12    14      6   4
> prop.table(t2)
         Type
Origin    Compact  Large Midsize  Small Sporty    Van
  USA      0.0753 0.1183  0.1075 0.0753 0.0860 0.0538
  non-USA  0.0968 0.0000  0.1290 0.1505 0.0645 0.0430
> margin.table(t3,1) #행
Origin
    USA non-USA 
     48      45 
> margin.table(t3,2) #열
Type
Compact   Large Midsize   Small  Sporty     Van 
     16      11      22      21      14       9 
> prop.table(t3)
         Type
Origin    Compact  Large Midsize  Small Sporty    Van
  USA      0.0753 0.1183  0.1075 0.0753 0.0860 0.0538
  non-USA  0.0968 0.0000  0.1290 0.1505 0.0645 0.0430
> addmargins(t3)
         Type
Origin    Compact Large Midsize Small Sporty Van Sum
  USA           7    11      10     7      8   5  48
  non-USA       9     0      12    14      6   4  45
  Sum          16    11      22    21     14   9  93
> addmargins(prop.table(t3))
         Type
Origin    Compact  Large Midsize  Small Sporty    Van    Sum
  USA      0.0753 0.1183  0.1075 0.0753 0.0860 0.0538 0.5161
  non-USA  0.0968 0.0000  0.1290 0.1505 0.0645 0.0430 0.4839
  Sum      0.1720 0.1183  0.2366 0.2258 0.1505 0.0968 1.0000
~~~

+ 빈도
***
# 수치자료

~~~R
> summary(Length)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  141.0   174.0   183.0   183.2   192.0   219.0 
> sd(Length)
[1] 14.60238
~~~

+ 평균
+ 표준편차
+ 중위수
***
# 수치자료 범주화 

~~~R
> grade=ifelse(airquality$Temp>=60,'상','하');grade
  [1] "상" "상" "상" "상" "하" "상" "상" "하" "상" "상" "상" "상" "상" "상" "하" "상" "상" "하" "상"
...
[153] "상"
> air0=data.frame(airquality,grade);air0
    Ozone Solar.R Wind Temp Month Day grade
1      41     190  7.4   67     5   1    상
2      36     118  8.0   72     5   2    상
3      12     149 12.6   74     5   3    상
4      18     313 11.5   62     5   4    상
5      NA      NA 14.3   56     5   5    하
6      28      NA 14.9   66     5   6    상
7      23     299  8.6   65     5   7    상
8      19      99 13.8   59     5   8    하
9       8      19 20.1   61     5   9    상
10     NA     194  8.6   69     5  10    상
...
> airquality$grade=ifelse(airquality$Temp>=60,'상','하');airquality
    Ozone Solar.R Wind Temp Month Day grade
1      41     190  7.4   67     5   1    상
2      36     118  8.0   72     5   2    상
3      12     149 12.6   74     5   3    상
4      18     313 11.5   62     5   4    상
5      NA      NA 14.3   56     5   5    하
6      28      NA 14.9   66     5   6    상
7      23     299  8.6   65     5   7    상
8      19      99 13.8   59     5   8    하
9       8      19 20.1   61     5   9    상
10     NA     194  8.6   69     5  10    상
...
> x = c(80, 88, 90, 93, 95, 94, 99, 78, 65)
> cat.x = (x >= 100) + (x < 90) + (x < 80)
> cat.x
[1] 1 1 0 0 0 0 0 2 2
> cat.x1 = factor(cat.x, labels = c("A", "B", "C"))
> cat.x1 #0~79/80~89/90~
[1] B B A A A A A C C
Levels: A B C
> cat.x2 = (x <= 100) + (x < 90) + (x < 80)
> cat.x2
[1] 2 2 1 1 1 1 1 3 3
> cat.x3 = factor(cat.x2, labels = c("A", "B", "C"))
> cat.x3
[1] B B A A A A A C C
Levels: A B C
> cat.x4 = cut(x, breaks = c(0, 80, 90, 100), include.lowest = T, right = F, labels = c("C", "B", "A"))
> cat.x4
[1] B B A A A A A C C
Levels: C B A
~~~
***
# 교차분석

> 범주자료 2개 - 두 범주형 자료간에 상호 관련성

## 검정통계량

<div style="overflow: auto;">

> $$\chi^2=\sum_{i}\sum_{j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}} \sim \chi^2_{(a-1)(b-1)}$$
</div>

+ a, b - 범주수

## 가설
> $$H_0 : 두\ 변수\ 독립$$

## 기대빈도

> 독립이게끔 하는 빈도, 차이가 없게끔 하는 빈도 - $E_{ij}$

## Example 1

<div style="overflow: auto;">

> $$H_0 : Origin과\ AirBags는\ 독립이다.$$
</div>

~~~R
> table(Origin,AirBags)
         AirBags
Origin    Driver & Passenger Driver only None
  USA                      9          23   16
  non-USA                  7          20   18
> t=xtabs(~Origin+AirBags,Cars93);t
         AirBags
Origin    Driver & Passenger Driver only None
  USA                      9          23   16
  non-USA                  7          20   18
> prop.table(t)
         AirBags
Origin    Driver & Passenger Driver only       None
  USA             0.09677419  0.24731183 0.17204301
  non-USA         0.07526882  0.21505376 0.19354839
> addmargins(t)
         AirBags
Origin    Driver & Passenger Driver only None Sum
  USA                      9          23   16  48
  non-USA                  7          20   18  45
  Sum                     16          43   34  93
> addmargins(prop.table(t))
         AirBags
Origin    Driver & Passenger Driver only       None        Sum
  USA             0.09677419  0.24731183 0.17204301 0.51612903
  non-USA         0.07526882  0.21505376 0.19354839 0.48387097
  Sum             0.17204301  0.46236559 0.36559140 1.00000000
> library(gmodels)
> CrossTable(Origin,AirBags,expected=T,chisq=T)

 
   Cell Contents
|-------------------------|
|                       N |
|              Expected N |
| Chi-square contribution |
|           N / Row Total |
|           N / Col Total |
| N / Table Total |
| --------------- |

 
Total Observations in Table:  93 

 
             | AirBags 
      | Origin        | Driver & Passenger   | Driver only          | None                 | Row Total            |
      | ------------- | -------------------- | -------------------- | -------------------- | -------------------- |
      | USA           | 9                    | 23                   | 16                   | 48                   |
      | 8.258         | 22.194               | 17.548               |                      |
      | 0.067         | 0.029                | 0.137                |                      |
      | 0.188         | 0.479                | 0.333                | 0.516                |
      | 0.562         | 0.535                | 0.471                |                      |
      | 0.097         | 0.247                | 0.172                |                      |
      | ------------- | -------------------- | -------------------- | -------------------- | -------------------- |
      | non-USA       | 7                    | 20                   | 18                   | 45                   |
      | 7.742         | 20.806               | 16.452               |                      |
      | 0.071         | 0.031                | 0.146                |                      |
      | 0.156         | 0.444                | 0.400                | 0.484                |
      | 0.438         | 0.465                | 0.529                |                      |
      | 0.075         | 0.215                | 0.194                |                      |
      | ------------- | -------------------- | -------------------- | -------------------- | -------------------- |
      | Column Total  | 16                   | 43                   | 34                   | 93                   |
      | 0.172         | 0.462                | 0.366                |                      |
      | ------------- | -------------------- | -------------------- | -------------------- | -------------------- |

 
Statistics for All Table Factors


Pearson's Chi-squared test 
------------------------------------------------------------
Chi^2 =  0.4806754     d.f. =  2     p =  0.7863623 
~~~

+ $p=0.786 \geq 0.05$이므로 $H_0$ 채택 - 독립

## Example 2

<div style="overflow: auto;">

> $$H_0 : Origin과\ Type은\ 독립이다.$$
</div>

~~~R
> CrossTable(Origin,Type,expected=T,chisq=T)

 
   Cell Contents
|-------------------------|
|                       N |
|              Expected N |
| Chi-square contribution |
|           N / Row Total |
|           N / Col Total |
| N / Table Total |
| --------------- |

 
Total Observations in Table:  93 

 
             | Type 
      | Origin        | Compact     | Large       | Midsize     | Small       | Sporty      | Van         | Row Total   |
      | ------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
      | USA           | 7           | 11          | 10          | 7           | 8           | 5           | 48          |
      | 8.258         | 5.677       | 11.355      | 10.839      | 7.226       | 4.645       |             |
      | 0.192         | 4.990       | 0.162       | 1.360       | 0.083       | 0.027       |             |
      | 0.146         | 0.229       | 0.208       | 0.146       | 0.167       | 0.104       | 0.516       |
      | 0.438         | 1.000       | 0.455       | 0.333       | 0.571       | 0.556       |             |
      | 0.075         | 0.118       | 0.108       | 0.075       | 0.086       | 0.054       |             |
      | ------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
      | non-USA       | 9           | 0           | 12          | 14          | 6           | 4           | 45          |
      | 7.742         | 5.323       | 10.645      | 10.161      | 6.774       | 4.355       |             |
      | 0.204         | 5.323       | 0.172       | 1.450       | 0.088       | 0.029       |             |
      | 0.200         | 0.000       | 0.267       | 0.311       | 0.133       | 0.089       | 0.484       |
      | 0.562         | 0.000       | 0.545       | 0.667       | 0.429       | 0.444       |             |
      | 0.097         | 0.000       | 0.129       | 0.151       | 0.065       | 0.043       |             |
      | ------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
      | Column Total  | 16          | 11          | 22          | 21          | 14          | 9           | 93          |
      | 0.172         | 0.118       | 0.237       | 0.226       | 0.151       | 0.097       |             |
      | ------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |

 
Statistics for All Table Factors


Pearson's Chi-squared test 
------------------------------------------------------------
Chi^2 =  14.07985     d.f. =  5     p =  0.01511005 


 
Warning message:
In chisq.test(t, correct = FALSE, ...) :
  Chi-squared approximation may be incorrect
~~~

+ $p=0.01511 \leq 0.05$이므로 $H_0$ 기각 - 독립 X
+ `Large` 항목의 경우 기대빈도와 5~6 정도 차이가 나는 것을 볼 수 있음
+ $\frac{Column\ Total\ N \times Row\ Total\ N}{Total\ N} = Expected\ N$
+ $\frac{(N-Expected\ N)^2}{Expected\ N} = Chi-square\ contribution$
+ 검정통계량 : $\chi^2=14.08$

***
# 상관분석

> 수치자료 2개

## 검정통계량
> $$T=r\sqrt{\frac{n-2}{1-r^2}}\sim t(n-2)$$

## 가설
> $$H_0 : \rho_{xy}=0$$

## 상관계수

<div style="overflow: auto;">

> $$r_{xy}=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}$$
</div>

## Example

> $$H_0 : \rho_{xy}=0$$

~~~R
> plot(Weight,EngineSize)
> r=cor(Weight,EngineSize);r
[1] 0.8450753
> t=r*sqrt((length(Weight)-2)/(1-r^2));t
[1] 15.07818
> (1-pt(t,length(Weight)-2))*2
[1] 0
~~~

+ $p=0 \leq 0.05$이므로 $H_0$ 기각 - $\rho_{xy}\neq0$
***
# 일표본 T-검정

> 수치자료 1개

## 검정통계량
> $$T=\frac{\bar{X}-\mu_0}{s/\sqrt{n}}\sim t(n-1)$$

## 가설
> $$H_0:\mu=\mu_0$$

+ `t.test()`
+ 문제에 맞게 양측 단측

## Example

~~~R
> mean(Price)
[1] 19.50968
> t.test(Price,mu=19,conf.level=0.95)

	One Sample t-test

data:  Price
t = 0.50884, df = 92, p-value = 0.6121
alternative hypothesis: true mean is not equal to 19
95 percent confidence interval:
 17.52034 21.49901
sample estimates:
mean of x 
 19.50968 

> t.test(Price,mu=25,conf.level=0.95)

	One Sample t-test

data:  Price
t = -5.4814, df = 92, p-value = 3.667e-07
alternative hypothesis: true mean is not equal to 25
95 percent confidence interval:
 17.52034 21.49901
sample estimates:
mean of x 
 19.50968 
~~~

+ $\mu=19$ - 귀무가설 채택($0.6121\geq 0.05$)
+ $\mu=25$ - 귀무가설 기각($3.667\times 10^-{7}\leq 0.05$)
***
# 독립표본 T-검정

> 집단 2개, 수치자료 1개 - 두 집단 평균비교

## 검정통계량

<div style="overflow: auto;">

> $$\sigma_1^2=\sigma_2^2\ -\ T=\frac{(\bar X_1 - \bar X_2)-(\mu_1 - \mu_2)}{\sqrt{S_p^2(1/n_1 + 1/n_2)}} \sim t(n_1+n_2-2)$$
> $$\sigma_1^2\neq\sigma_2^2\ -\ T=\frac{(\bar X_1 - \bar X_2)-(\mu_1 - \mu_2)}{\sqrt{S_1^2/n_1 + S_2^2/n_2}} \sim t(u^*)$$
</div>

+ 가중평균 : 집단의 개수가 다르면 가중치를 대입하여 평균 - $n_1 \bar x_1 + n_2 \bar x_2 \over n_1 + n_2$
+ 합동분산 $S_p^2 = \frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{(n_1-1)+(n_2-1)}$ - S는 자유도가 1
+ $u^*$는 실수

## 가설
> $$H_0 : \mu_1-\mu_2=0$$

## 등분산 검정

### 가설
> $$H_0 : \frac{\sigma_1^2}{\sigma_2^2}=1,\ \sigma_1^2=\sigma_2^2$$

+ 분산의 비는 F분포
+ 우단측 검정만 실행
+ 큰 $\sigma$를 위에

## Example 1

> 등분산 검정
> $$H_0 : \frac{\sigma_1^2}{\sigma_2^2}=1$$

~~~R
> var.test(mpg~am,mtcars)

	F test to compare two variances

data:  mpg by am
F = 0.38656, num df = 18, denom df = 12, p-value = 0.06691
alternative hypothesis: true ratio of variances is not equal to 1
95 percent confidence interval:
 0.1243721 1.0703429
sample estimates:
ratio of variances 
         0.3865615 
~~~

+ `var.test(수치~범주,Data)`
+ $p=0.06691 \geq 0.05$이므로 귀무가설 채택
+ $\sigma_1^2=\sigma_2^2$

> 독립표본 T-검정(등분산)
> $$H_0 : \mu_1-\mu_2=0$$

~~~R
> t.test(mpg~am,mtcars,var.equal=T)

	Two Sample t-test

data:  mpg by am
t = -4.1061, df = 30, p-value = 0.000285
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -10.84837  -3.64151
sample estimates:
mean in group 0 mean in group 1 
       17.14737        24.39231 
~~~

+ $p=0.000285 \leq 0.05$이므로 귀무가설 기각
+ $\mu_1 \neq \mu_2$

> 해석

+ `group 1`의 평균은 24.39231, `group 0`의 평균은 17.14737으로 `group 1`의 평균이 더 크다.

## Example 2

> 등분산 검정
> $$H_0 : \frac{\sigma_1^2}{\sigma_2^2}=1$$

~~~R
> var.test(Price~Origin,Cars93)

	F test to compare two variances

data:  Price by Origin
F = 0.47796, num df = 47, denom df = 44, p-value = 0.01387
alternative hypothesis: true ratio of variances is not equal to 1
95 percent confidence interval:
 0.2645004 0.8587304
sample estimates:
ratio of variances 
         0.4779637 
~~~

+ $p=0.01387 \leq 0.05$이므로 귀무가설 기각
+ $\sigma_1^2 \neq \sigma_2^2$

> 독립표본 T-검정(이분산)
> $$H_0 : \mu_1-\mu_2=0$$

~~~R
> t.test(Price~Origin,Cars93,var.equal=F)

	Welch Two Sample t-test

data:  Price by Origin
t = -0.95449, df = 77.667, p-value = 0.3428
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -5.974255  2.102311
sample estimates:
    mean in group USA mean in group non-USA 
             18.57292              20.50889 
~~~

+ $p=0.3428 \geq 0.05$이므로 귀무가설 채택
+ $\mu_1=\mu_2$

> 해석

+ `USA`와 `non-USA`의 `Price`차이는 없다.

## Example 3

~~~R
> var.test(Weight~Origin,Cars93)

	F test to compare two variances

data:  Weight by Origin
F = 0.90622, num df = 47, denom df = 44, p-value = 0.7388
alternative hypothesis: true ratio of variances is not equal to 1
95 percent confidence interval:
 0.501495 1.628160
sample estimates:
ratio of variances 
         0.9062231 

> t.test(Weight~Origin,Cars93)

	Welch Two Sample t-test

data:  Weight by Origin
t = 2.1016, df = 89.825, p-value = 0.03839
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
  13.82668 492.13165
sample estimates:
    mean in group USA mean in group non-USA 
             3195.312              2942.333 
~~~

> 해석

+ `USA`의 차가 `Weight` 즉, 무게가 `non-USA`의 차보다 더 나간다.

***
# 대응표본 T-검정

> 수치자료 2개 - 전후비교

## 검정통계량

<div style="overflow: auto;">

> $$T=\frac{\bar D - \mu_D}{S_D/\sqrt{n}} \sim t(n-1)$$
</div>

## 가설

<div style="overflow: auto;">

> $$H_0 : \mu_{before}-\mu_{after}=\mu_D=0$$
</div>

## Example

<div style="overflow: auto;">

> $$H_0 : \mu_{before}-\mu_{after}=\mu_D=0$$
</div>

~~~R
> attach(shoes)
> t.test(A,B,paired=T)

	Paired t-test

data:  A and B
t = -3.3489, df = 9, p-value = 0.008539
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -0.6869539 -0.1330461
sample estimates:
mean of the differences 
                  -0.41 

> t.test(B,A,paired=T)

	Paired t-test

data:  B and A
t = 3.3489, df = 9, p-value = 0.008539
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 0.1330461 0.6869539
sample estimates:
mean of the differences 
                   0.41 

> D=A-B
> t.test(D,mu=0)

	One Sample t-test

data:  D
t = -3.3489, df = 9, p-value = 0.008539
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 -0.6869539 -0.1330461
sample estimates:
mean of x 
    -0.41
 
> mean(A)
[1] 10.63
> mean(B)
[1] 11.04
~~~

+ $p=0.008539\leq0.05$이므로 귀무가설 기각
+ $\mu_A\neq \mu_B$

> 해석

+ 평균적으로 `A`가 `B`보다 0.41 작다.

***
# 일원배치 분산분석

> 집단 N개, 수치자료 1개 - N개 집단 평균비교($N \geq 3$)

## 검정통계량
> $$F=\frac{S_1^2}{S_2^2}=\frac{MSB}{MSW}$$

## 가설

<div style="overflow: auto;">

> $$H_0 : \mu_1 = \mu_2 = ... = \mu_k$$
> $$H_1 : \mu_j(적어도\ 하나)는\ 같지\ 않다$$
</div>

## 분산

> $$S^2=\frac{1}{n-1}\sum(x_i-\bar x)^2$$
+ 편차의 제곱합
+ 자유도로 나눔
+ 평균제곱합

> $$Y_{ij}$$
+ i번째 집단 j번째 Data

<div style="overflow: auto;">

$$Y_{ij}-\bar Y=(\bar Y_i-\bar Y)+(Y_{ij}-\bar Y_i)$$
</div>
<div style="overflow: auto;">

$$(Y_{ij}-\bar Y)^2=(\bar Y_i-\bar Y)^2+(Y_{ij}-\bar Y_i)^2$$
</div>
<div style="overflow: auto;">

$$\sum_i \sum_j (Y_{ij}-\bar Y)^2=\sum_i \sum_j (\bar Y_i-\bar Y)^2+\sum_i \sum_j (Y_{ij}-\bar Y_i)^2$$
</div>
<div style="overflow: auto;">

$$총제곱합(Sum\ of\ Square\ Total)=집단간\ 제곱합(Sum\ of\ Square\ Between)+집단내\ 제곱합(Sum\ of\ Square\ Within)$$
</div>
<div style="overflow: auto;">

$$df : n-1=(k-1)+(n-k)$$
</div>

+ `k` - 집단수

## 다중비교

+ 귀무가설을 기각할때 차이가 있는지 여부만 알고 어떻게 차이가 있는지 알지 못함
+ 따라서 다중비교, 하지만 분산분석과는 별개

> Tukey - HSD(Honest Significant Difference) 방법
+ 가장 보수적

## 분산분석표 분석

+ SSB
+ SSW
+ df

## Example

> $$H_0 : \mu_1 = \mu_2 = ... = \mu_k$$

~~~R
> a=lm(Price~DriveTrain,Cars93);a

Call:
lm(formula = Price ~ DriveTrain, data = Cars93)

Coefficients:
    (Intercept)  DriveTrainFront   DriveTrainRear  
       17.63000         -0.09418         11.32000  

> anova(a)
Analysis of Variance Table

Response: Price
           Df Sum Sq Mean Sq F value    Pr(>F)    
DriveTrain  2 1722.3  861.14  11.295 4.202e-05 ***
Residuals  90 6861.7   76.24                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
~~~

+ `DriveTrain` 집단간, `Residual` 집단내
+ $p=4.202 \times 10^{-5} \leq 0.05$이므로 귀무가설 기각
+ 각 집단의 평균이 차이를 보임

> 다중비교

~~~R
> a1=aov(Price~DriveTrain,Cars93);a1
Call:
   aov(formula = Price ~ DriveTrain, data = Cars93)

Terms:
                DriveTrain Residuals
Sum of Squares    1722.286  6861.735
Deg. of Freedom          2        90

Residual standard error: 8.731638
Estimated effects may be unbalanced
> TukeyHSD(a1)
  Tukey multiple comparisons of means
    95% family-wise confidence level

Fit: aov(formula = Price ~ DriveTrain, data = Cars93)

$DriveTrain
                 diff       lwr       upr     p adj
Front-4WD  -0.0941791 -7.148353  6.959995 0.9994421
Rear-4WD   11.3200000  2.931875 19.708125 0.0050886
Rear-Front 11.4141791  5.624162 17.204197 0.0000278

> aggregate(Price,by=list(DriveTrain),mean)
  Group.1        x
1     4WD 17.63000
2   Front 17.53582
3    Rear 28.95000
~~~

> 해석

+ 후륜구동 자동차가 4륜, 전륜에 비해 비싼 가격을 가지는 경향을 보임
+ 4륜과 전륜 자동차는 비슷한 가격을 가지는 경향을 보임
***
# 회귀분석

<div style="overflow: auto;">

> $$Y_i(종속)=\beta_0+\beta_1x_i(독립)+\epsilon_i$$
> $$\epsilon_i(오차항) \sim N(0,\sigma^2) - 확률변수$$
</div>

+ 오차항(회귀분석은 아래 가정을 따라야 함)
  + 정규성
  + 독립성
  + 등분산성
+ 독립변수 -> 종속변수 영향력? - $\beta_1$
+ $\beta_0, \beta_1$ - 모수(상수)
+ $E(Y_i)=\beta_0+\beta_1x_i$
+ $V(Y_i)=\sigma^2$

> $$Y_i \sim N(\beta_0+\beta_1x_i,\sigma^2)$$

<div style="overflow: auto;">

> $$D=\sum_{i=1}^ne_i=\sum_{i=1}^n(Y_i-\hat{Y_i})^2=\sum_{i=1}^n(Y_i-y(x_i))^2$$
</div>

+ $e_i$ - Residual(잔차)
+ $\hat{Y_i}$ - 추정량(예측치)

## 검정통계량

### F-검정

<div style="overflow: auto;">

$$MSR=\frac{SSR}{k},\ MSE=\frac{SSE}{n-k-1}$$
</div>
<div style="overflow: auto;">

> $$F=\frac{MSR}{MSE}$$
</div>

### T-검정

<div style="overflow: auto;">

> $$T=\frac{\hat{\beta_1}-\beta_1}{\sqrt{\frac{MSE}{\sum(x_i-\bar{x})^2}}} \sim t(n-2)$$
</div>

+ `F`는 모든 계수 검정, `T`는 $\beta_1$만 검정

## 가설

### F-검정

<div style="overflow: auto;">

> $$H_0 : \beta_1 = 0(회귀모형\ 적합\ X)$$
</div>

### T-검정

<div style="overflow: auto;">

> $$H_0 : \beta_0=0, T=\frac{\hat{\beta_0}-\beta_0}{\sqrt{MSE(\frac{1}{n}+\frac{\bar{x}^2}{\sum(x_i-\bar{x})^2})}}\sim t(n-2)$$
> $$H_0 : \beta_1=0, T=\frac{\hat{\beta_1}-\beta_1}{\sqrt{\frac{MSE}{\sum(x_i-\bar{x})^2}}} \sim t(n-2)$$
</div>

## Least squares regression
> $$\frac{\partial D}{\partial \hat \beta_0}=0$$
> $$\frac{\partial D}{\partial \hat \beta_1}=0$$

## $\hat \beta_0,\ \hat \beta_1$의 분포

<div style="overflow: auto;">

> $$\hat{\beta_0}=\bar{Y}-\hat{\beta_1}\bar{x} \sim N(\beta_0,\sigma^2(\frac{1}{n}+\frac{\bar{x}^2}{\sum(x_i-\bar{x})^2})$$
> $$\hat{\beta_1}=\frac{\sum(x_i-\bar{x})(Y_i-\bar{Y})}{\sum(x_i-\bar{x})^2} \sim N(\beta_1,\frac{\sigma^2}{\sum(x_i-\bar{x})^2})$$
</div>

## 분산분석

<div style="overflow: auto;">

$$Y_{i}-\bar Y=(\hat Y_i-\bar Y)+(Y_{i}-\hat Y_i)$$
</div>
<div style="overflow: auto;">

$$\sum (Y_{i}-\bar Y)^2=\sum (\hat Y_i-\bar Y)^2+\sum (Y_{i}-\hat Y_i)^2$$
</div>
<div style="overflow: auto;">

$$총제곱합(SST)=회귀제곱합(SSR)+잔(오)차제곱합(SSE)$$
</div>
<div style="overflow: auto;">

$$df : n-1=(k)+(n-k-1)$$
</div>

+ `SSR`은 크고 `SSE`는 작아야 유리
+ `k` - 독립변수 개수

## $R^2$ - 결정계수

> $$R^2=\frac{SSR}{SST}$$

+ 설명력
+ 높을수록 좋음

## 진행 순서

1. `plot(param1,param2)` - 산점도 확인
2. `name=lm(param2~param1,Data)` - Coefficient 확인
3. `anova(name)` - F, SSR, SSE(Residual) 확인
4. `summary(name)` - 결정계수, T 확인

## Example 1

> $$H_0 : \beta_1 = 0(회귀모형\ 적합\ X)$$

~~~R
> a=lm(Price~Length,Cars93);a

Call:
lm(formula = Price ~ Length, data = Cars93)

Coefficients:
(Intercept)       Length  
   -41.5246       0.3331  

> anova(a)
Analysis of Variance Table

Response: Price
          Df Sum Sq Mean Sq F value    Pr(>F)    
Length     1 2177.3  2177.3  30.925 2.663e-07 ***
Residuals 91 6406.8    70.4                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
> summary(a)

Call:
lm(formula = Price ~ Length, data = Cars93)

Residuals:
    Min      1Q  Median      3Q     Max 
-10.969  -5.708  -2.674   2.790  41.126 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept) -41.52458   11.00974  -3.772 0.000288 ***
Length        0.33315    0.05991   5.561 2.66e-07 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 8.391 on 91 degrees of freedom
Multiple R-squared:  0.2536,	Adjusted R-squared:  0.2454 
F-statistic: 30.93 on 1 and 91 DF,  p-value: 2.663e-07

> 2177/(6406+2177)
[1] 0.2536409
~~~

+ Length - `SSR`
+ Residual - `SSE`
+ $2.663 \times 10^{-7} \leq 0.05$이므로 귀무가설 기각
+ $\beta_1 \neq 0$
+ `Length`의 `F value`를 제곱하면 `t vlue`
+ 약 25% 설명

## Example 2

<div style="overflow: auto;">

> $$H_0 : \beta_1 = 0(회귀모형\ 적합\ X)$$
</div>

~~~R
> plot(Price,Weight)
> a=lm(Weight~Price,Cars93);a

Call:
lm(formula = Weight ~ Price, data = Cars93)

Coefficients:
(Intercept)        Price  
    2301.82        39.52  

> anova(a)
Analysis of Variance Table

Response: Weight
          Df   Sum Sq  Mean Sq F value    Pr(>F)    
Price      1 13408751 13408751  65.584 2.395e-12 ***
Residuals 91 18605215   204453                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
> summary(a)

Call:
lm(formula = Weight ~ Price, data = Cars93)

Residuals:
     Min       1Q   Median       3Q      Max 
-1223.29  -304.86   -56.05   238.62  1067.10 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  2301.82     106.13  21.688  < 2e-16 ***
Price          39.52       4.88   8.098  2.4e-12 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 452.2 on 91 degrees of freedom
Multiple R-squared:  0.4188,	Adjusted R-squared:  0.4125 
F-statistic: 65.58 on 1 and 91 DF,  p-value: 2.395e-12

> abline(a=2301,b=39.5,col='Red')
~~~

+ Price - `SSR`
+ Residual - `SSE`
+ $2.395 \times 10^{-12} \leq 0.05$이므로 귀무가설 기각
+ $\beta_1 \neq 0$
+ 약 42% 설명

<img src="/images/r-final-term/plot-1.png" alt="plot-1" width="1145" />

## Example 3

> $$H_0 : \beta_1 = 0(회귀모형\ 적합\ X)$$

~~~R
> plot(Price,Max.Price)
> a=lm(Max.Price~Price,Cars93);a

Call:
lm(formula = Max.Price ~ Price, data = Cars93)

Coefficients:
(Intercept)        Price  
    0.03048      1.12090  

> anova(a)
Analysis of Variance Table

Response: Max.Price
          Df  Sum Sq Mean Sq F value    Pr(>F)    
Price      1 10785.2 10785.2  2402.1 < 2.2e-16 ***
Residuals 91   408.6     4.5                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
> summary(a)

Call:
lm(formula = Max.Price ~ Price, data = Cars93)

Residuals:
    Min      1Q  Median      3Q     Max 
-3.9598 -1.1627 -0.1561  0.8119 10.5857 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  0.03048    0.49736   0.061    0.951    
Price        1.12090    0.02287  49.012   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 2.119 on 91 degrees of freedom
Multiple R-squared:  0.9635,	Adjusted R-squared:  0.9631 
F-statistic:  2402 on 1 and 91 DF,  p-value: < 2.2e-16

> abline(a,col='red')
~~~

+ Price - `SSR`
+ Residual - `SSE`
+ $2.2 \times 10^{-16} \leq 0.05$이므로 귀무가설 기각
+ $\beta_1 \neq 0$
+ 약 96% 설명

<img src="/images/r-final-term/plot-2.png" alt="plot-2" width="1145" />

# Finale

~~~R
go=function(a){
	if(a==1){
		print('gyocha, label 2')
		print('H_0 : var independent')
		print('gumjung : chi^2=sumsum(O-E)^2/E~chi^2(a-1)(b-1)')
		print('library(gmodels)')
		print('CrossTable(var1,var2,expected=T,chisq=T)')
	}else if(a==2){
		print('sanggwan, num 2')
		print('H_0 : rho_xy=0')
		print('r=sum(xi-xbar)(yi-ybar)/sqrt(sum(xi-xbar)^2*sum(yi-ybar)^2)')
		print('gumjung : T=r*sqrt((length(var1)-2)/(1-r^2))~t(n-2)')
		print('r=cor(var1,var2)')
		print('t=r*sqrt((length(var1)-2)/(1-r^2))')
		print('(1-pt(t,length(var1)-2))*2')
	}else if(a==3){
		print('1pyobon T-gumjung, num 1')
		print('H_0 : mu=mu_0')
		print('gumjung : T=(Xbar-mu0)/(S/sqrt(n))~t(n-1)')
		print('t.test(var1,mu=mu_0)')
	}else if(a==4){
		print('dokrib T-gumjung, num 1(label 2)')
		print('H_0 : mu_1-mu_2=0')
		print('gumjung : daeeung, Normalize - eq~t(n1+n2-2), neq~t(u*)')
		print('H_0 : sigma_1^2/sigma_2^2=1')
		print('var.test(num~lab,Data)')
		print('t.test(num~lab,Data,var,equal=T or F)')
	}else if(a==5){
		print('daeeung T-gumjung, num 2(be af)')
		print('H_0 : mu_before-mu_after=mu_D=0')
		print('gumjung : T=(Dbar-muD)/(S_D/sqrt(n))~t(n-1)')
		print('t.test(var1,var2,paired=T)')
		print('D=var1-var2')
		print('t.test(D,mu=0)')
	}else if(a==6){
		print('ilwonbatch bunsanbunsuck, num 1(label 3up)')
		print('H_0=mu_1=mu_2=...=mu_k')
		print('gumjung : F=S1^2/S2^2=MSB/MSW')
		print('Yij-Ybar=Yibar-Ybar+Yij-Yibar')
		print('SST=SSB+SSW')
		print('n-1=k-1+n-k')
		print('name1=lm(num~lab,Data)')
		print('anova(name1)')
		print('name2=aov(num~lab,Data)')
		print('TukeyHSD(name2)')
		print('aggregate(num,by=list(lab),mean)')
	}else if(a==7){
		print('reg')
		print('H_0 : hatbeta1=0')
		print('gumjung : F=MSR/MSE, T=(beta1hat-beta1)/sqrt(MSE/sum(xi-xbar)^2)~t(n-2)')
		print('Yi-Ybar=Yihat-Ybar+Yi-Yihat')
		print('SST=SSR+SSE')
		print('n-1=k+(n-k-1)')
		print('R^2=SSR/SST')
		print('plot(var1,var2)')
		print('name=lm(num~lab,Data)')
		print('anova(name)')
		print('summary(name)')
	}else if(a==8){
		print('Data chisa')
		print('name[which(jogun),c()]')
		print('subset(name,select=(),subset=c())')
	}else if(a==9){
		print('r-chuchul,d-hwakrule,p-nujuk,q-bunwesu')
		print('binom,geom,nbinom,pois,hyper')
		print('dhyper(2,3,2,3) # B=3, W=2, 3gae chuchul, B=2, W=1')
	}else if(a==10){
		print('table,with,xtabs,prop.table,margin.table,addmargins')
		print('cut(x,breaks=c(),include.lowest=,right=,labels=c())')
		print('factor(x,labels=c())')
	}else if(a==11){
		print('beta0hat~N(beta0,sigma^2(1/n+xbar^2/sum(xi-xbar)))')
		print('beta1hat~N(beta1,sigma^2/sum(xi-xbar))')
		print('확률변수 - 표본공간의 각 사건을 수치로 대응시켜주는 함수')
	}
}
~~~