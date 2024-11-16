---
title: Normality Test
date: 2020-01-10 13:57:42
categories:
- Etc.
tags:
- R
- Statistics
- B.S. Course Work
---
# 정규성 검정 function

## shapiro.test(), qqnorm(), qqline()

~~~R
> library(MASS)
> attach(Cars93)
> shapiro.test(Price)

	Shapiro-Wilk normality test

data:  Price
W = 0.88051, p-value = 4.235e-07

> str(Price)
 num [1:93] 15.9 33.9 29.1 37.7 30 15.7 20.8 23.7 26.3 34.7 ...
> qqnorm(Price)
> qqline(Price)
> qqnorm(log(Price))
> qqline(log(Price))
> shapiro.test(log(Price))

	Shapiro-Wilk normality test

data:  log(Price)
W = 0.9841, p-value = 0.32
~~~
<!-- more -->
+ 정규분포를 따르지 않으면 sample 수가 적음
+ `shaprio.test()` - $H_0$ : 정규분포(정규성 분포)
+ `shapiro.test(Price)` : $p-value \leq 0.05$ 이므로 귀무가설 기각
+ `shapiro.test(log(Price))` : $p-value \geq 0.05$ 이므로 귀무가설 채택

<img width="1145" alt="qqnorm(Price)" src="https://user-images.githubusercontent.com/42334717/72127505-40f23880-33b3-11ea-8a4d-f2d353bf83a6.png">
<img width="1145" alt="qqline(Price)" src="https://user-images.githubusercontent.com/42334717/72127550-6717d880-33b3-11ea-8bde-de8dcac9ba4f.png">

+ 선에 맞춰서 몰려져있어야 정규분포

<img width="1145" alt="qqnorm(log(Price)), qqline(log(Price))" src="https://user-images.githubusercontent.com/42334717/72127654-b827cc80-33b3-11ea-90f2-535d5199d6e9.png">

+ p-value $\leq$ 0.05 이므로 귀무가설 채택

## xtabs(), table(), prop.table()

~~~R
> str(Cars93)
'data.frame':	93 obs. of  27 variables:
 $ Manufacturer      : Factor w/ 32 levels "Acura","Audi",..: 1 1 2 2 3 4 4 4 4 5 ...
...
 $ Make              : Factor w/ 93 levels "Acura Integra",..: 1 2 4 3 5 6 7 9 8 10 ...
> t1=table(Origin);t1
Origin
    USA non-USA 
     48      45 
> xtabs(~Origin,Cars93)
Origin
    USA non-USA 
     48      45 
> prop.table(table(Origin))
Origin
     USA  non-USA 
0.516129 0.483871 
> prop.table(t1)
Origin
     USA  non-USA 
0.516129 0.483871 
> options('digits'=3)
> prop.table(table(Origin))
Origin
    USA non-USA 
  0.516   0.484 
> prop.table(t1)
Origin
    USA non-USA 
  0.516   0.484
~~~

+ `~`다음은 보통 범주형 자료
+ `xtabs()` - `attach()`없어도 가능
+ `prop.table()` - 비율 출력
+ `options('digits'=n)` - 소수 자릿수 수정
  
## with() - 교차표(범주형 2개)

~~~R
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

+ `with()`, `xtabs()` - 교차표 생성
+ `margin.table()` - table의 행 or 열 출력
+ `addmargins()` - 합계 출력

***
# 교차분석(범주형 2개) - CrossTable()

## 검정통계량

<div style="overflow: auto;">

> $$\chi^2=\sum_{i}\sum_{j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}} \sim \chi^2_{(a-1)(b-1)}$$
</div>

+ a, b - 범주수

## 가설
<div style="overflow: auto;">

> $$H_0 : 두\ 변수\ 독립\ O,\ H_1 : 두\ 변수\ 독립\ X$$
</div>

~~~R
> library(gmodels)
> CrossTable(Origin,Type)

 
   Cell Contents
|-------------------------|
|                       N |
| Chi-square contribution |
|           N / Row Total |
|           N / Col Total |
|         N / Table Total |
|-------------------------|

 
Total Observations in Table:  93 

 
             | Type 
      Origin |   Compact |     Large |   Midsize |     Small |    Sporty |       Van | Row Total | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
         USA |         7 |        11 |        10 |         7 |         8 |         5 |        48 | 
             |     0.192 |     4.990 |     0.162 |     1.360 |     0.083 |     0.027 |           | 
             |     0.146 |     0.229 |     0.208 |     0.146 |     0.167 |     0.104 |     0.516 | 
             |     0.438 |     1.000 |     0.455 |     0.333 |     0.571 |     0.556 |           | 
             |     0.075 |     0.118 |     0.108 |     0.075 |     0.086 |     0.054 |           | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
     non-USA |         9 |         0 |        12 |        14 |         6 |         4 |        45 | 
             |     0.204 |     5.323 |     0.172 |     1.450 |     0.088 |     0.029 |           | 
             |     0.200 |     0.000 |     0.267 |     0.311 |     0.133 |     0.089 |     0.484 | 
             |     0.562 |     0.000 |     0.545 |     0.667 |     0.429 |     0.444 |           | 
             |     0.097 |     0.000 |     0.129 |     0.151 |     0.065 |     0.043 |           | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
Column Total |        16 |        11 |        22 |        21 |        14 |         9 |        93 | 
             |     0.172 |     0.118 |     0.237 |     0.226 |     0.151 |     0.097 |           | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
> CrossTable(Origin,Type,expected=T,chisq=T)

 
   Cell Contents
|-------------------------|
|                       N |
|              Expected N |
| Chi-square contribution |
|           N / Row Total |
|           N / Col Total |
|         N / Table Total |
|-------------------------|

 
Total Observations in Table:  93 

 
             | Type 
      Origin |   Compact |     Large |   Midsize |     Small |    Sporty |       Van | Row Total | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
         USA |         7 |        11 |        10 |         7 |         8 |         5 |        48 | 
             |     8.258 |     5.677 |    11.355 |    10.839 |     7.226 |     4.645 |           | 
             |     0.192 |     4.990 |     0.162 |     1.360 |     0.083 |     0.027 |           | 
             |     0.146 |     0.229 |     0.208 |     0.146 |     0.167 |     0.104 |     0.516 | 
             |     0.438 |     1.000 |     0.455 |     0.333 |     0.571 |     0.556 |           | 
             |     0.075 |     0.118 |     0.108 |     0.075 |     0.086 |     0.054 |           | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
     non-USA |         9 |         0 |        12 |        14 |         6 |         4 |        45 | 
             |     7.742 |     5.323 |    10.645 |    10.161 |     6.774 |     4.355 |           | 
             |     0.204 |     5.323 |     0.172 |     1.450 |     0.088 |     0.029 |           | 
             |     0.200 |     0.000 |     0.267 |     0.311 |     0.133 |     0.089 |     0.484 | 
             |     0.562 |     0.000 |     0.545 |     0.667 |     0.429 |     0.444 |           | 
             |     0.097 |     0.000 |     0.129 |     0.151 |     0.065 |     0.043 |           | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
Column Total |        16 |        11 |        22 |        21 |        14 |         9 |        93 | 
             |     0.172 |     0.118 |     0.237 |     0.226 |     0.151 |     0.097 |           | 
-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|

 
Statistics for All Table Factors


Pearson's Chi-squared test 
------------------------------------------------------------
Chi^2 =  14.1     d.f. =  5     p =  0.0151 


 
Warning message:
In chisq.test(t, correct = FALSE, ...) :
  Chi-squared approximation may be incorrect
~~~

+ N, Expected N, 각 셀 chi-square, 행%, 열%, 전체%에서 몇 %를 차지하는지
+ `expected=T` - 기대빈도
+ `chisq=T` - 카이제곱분포(표준정규분포 제곱)
+ $16 \times 48 \over 93$
+ $(7-8.258)^2 \over 8.258$
+ 유의수준 0.05 $\geq$ 유의확률 0.015 - 귀무가설 기각
+ `Origin`에 따라서 `Type`의 차이 O
+ 행%를 보고 해석 - 핵심적이고 큰 차이가 나는 것을 기술(ex. `Large`)

## ex) 두 변수 독립?

+ $H_0$ : 두 변수 독립(Origin에 따라서 Type 차이 X)
+ $H_1$ : 두 변수 독립 X

||O|X|sum|
|:-:|:-:|:-:|:-:|
|남|100|100|200|
|여|50|50|100|
|sum|150|150|300|

> 두 사상이 독립(사건의 독립) $$p(O|남)=\frac{p(O \cap 남)}{p(남)}$$

+ 기대빈도 : 독립이게끔 하는 빈도, 차이가 없게끔 하는 빈도 - $E_{ij}$
+ 관측빈도 - $O_{ij}$

<div style="overflow: auto;">

> $$\chi^2=\sum_{i}\sum_{j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}} \sim \chi^2_{(a-1)(b-1)}$$
</div>

+ `a`, `b` - 범주수

***
# 상관분석(수치형 2개)

## 검정통계량
> $$T=r\sqrt{\frac{n-2}{1-r^2}}\sim t(n-2)$$

## 가설
> $$H_0 : \rho_{xy}=0,\ H_1 : \rho_{xy}\neq0$$

~~~R
> plot(Width,Length)
> r=cor(Width,Length);r
[1] 0.822
~~~

<img width="1145" alt="산점도" src="https://user-images.githubusercontent.com/42334717/72130050-b3671680-33bb-11ea-89f5-03718cf943af.png">

## 상관계수 

<div style="overflow: auto;">

> 두 수치변수의 직선적인 정도 - 공분산/분산(무차원수)
> $$r_{xy}=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}$$
</div>

> 공분산 $$\sum{(x_i-\bar{x})(y_i-\bar{y})}\over n-1$$

> 표본분산 $$s^2=\frac{\sum{(x_i-\bar{x})^2}}{n-1}$$

+ 곡선형은 상관계수 무의미
+ 산점도 그리고 상관계수
+ $r_{xy}$ - 표본상관계수
+ $\rho_{xy}$ - 모상관계수

## t분포

$$T=r\sqrt{\frac{n-2}{1-r^2}}\sim t(n-2)$$

+ 표준정규분포보다 꼬리가 두꺼움
+ $\sigma \over \sqrt{n}$대신 `s`를 사용하여 자유도 -1 : n-1
+ $Z_{0.025}=1.96 \leq t_{0.025}$

~~~R
> qnorm(0.95)
[1] 1.64
> qnorm(0.975)
[1] 1.96
> qnorm(0.995)
[1] 2.58
~~~

+ `df` - degree of freedom

> 유의확률을 수식으로

~~~R
> t=r*sqrt((length(Width)-2)/(1-r^2));t #검정통계량
[1] 13.8
> (1-pt(t,length(Width)-2))*2 #유의확률
[1] 0
~~~

***
# 일표본 T-검정 - t.test()

> 평균비교

## 검정통계량
> $$T=\frac{\bar{X}-\mu_0}{s/\sqrt{n}}\sim t(n-1)$$

## 가설
> $$H_0:\mu=\mu_0$$

## Example

~~~R
t.test(x,alternative=c('two.sided','less','greater'),mu=175,conf.level=0.95)
> t.test(OBP,mu=0.33,conf.level=0.95)

	One Sample t-test

data:  OBP
t = -0.168458, df = 437, p-value = 0.8663
alternative hypothesis: true mean is not equal to 0.33
95 percent confidence interval:
 0.32571002 0.33361264
sample estimates:
 mean of x 
0.32966133 

> t=(mean(OBP)-0.33)/(sd(OBP)/sqrt(length(OBP)));t
[1] -0.16845758
> pt(t,437)*2 #유의확률
[1] 0.86630125
~~~

+ `alternative` 안쳐도 됨, Default 양측
+ `less`는 좌단측 `greater`는 우단측
+ `x`는 변수명
+ `config.level` : 1-$\alpha$(유의수준)
+ 위의 값 `OBP`는 귀무가설 채택

> 검정통계량 : 양수 - `1-pt()` / 음수 - `pt()`

***
# 독립표본 T-검정

<div style="overflow: auto;">

> 모수 $$\mu_1-\mu_2 \ -> \bar X_1 - \bar X_2 \sim N(\mu_1 - \mu_2, \sigma_1^2/n_1 + \sigma_2^2/n_2)$$
</div>

<div style="overflow: auto;">

> $$Z=\frac{(\bar X_1 - \bar X_2)-(\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} \sim N(0,1^2)$$
</div>

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

## 등분산 검정 - F분포(표본분산)

### 가설
> $$H_0 : \frac{\sigma_1^2}{\sigma_2^2}=1,\ \sigma_1^2=\sigma_2^2$$

+ 분산의 비는 F분포
+ 우단측 검정만 실행
+ 큰 $\sigma$를 위에

~~~R
> 1-pt(1.96,100)
[1] 0.02638945
> 1-pnorm(1.96)
[1] 0.0249979
~~~

+ 실제로는 보통 t분포를 사용함

## Example

~~~R
> summary(am)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 0.0000  0.0000  0.0000  0.4062  1.0000  1.0000 
> summary(mpg)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  10.40   15.43   19.20   20.09   22.80   33.90 
~~~

+ 귀무가설 : am에 따라서 mpg의 평균 차이가 없다

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

+ `var.test()` : 등분산 검정 함수(수치~범주)
+ $p=0.06691>0.05$ : 채택 - 등분산

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

+ 등분산일 경우만 `var.equal=T`, 이분산은 `default`
+ $p=0.000285<0.05$ : 기각 - 차이가 있다
+ 양측검정(0이 아니다) - 단측검정이면 유의확률/2
+ 1번 집단의 mpg가 더 크다고 말할 수 있음

~~~R
> t.test(mpg~am,mtcars) # 자유도 소수

	Welch Two Sample t-test

data:  mpg by am
t = -3.7671, df = 18.332, p-value = 0.001374
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -11.280194  -3.209684
sample estimates:
mean in group 0 mean in group 1 
       17.14737        24.39231 

> var.test(mpg~vs) # 이것도 등분산

	F test to compare two variances

data:  mpg by vs
F = 0.51515, num df = 17, denom df = 13, p-value = 0.1997
alternative hypothesis: true ratio of variances is not equal to 1
95 percent confidence interval:
 0.1714935 1.4353527
sample estimates:
ratio of variances 
         0.5151485 
~~~

`성별에 따라 학점 차이?` - 등분산 검정
***
# 대응표본 T-검정(전후 수치형 2개)

## 검정통계량

<div style="overflow: auto;">

> $$T=\frac{\bar D - \mu_D}{S_D/\sqrt{n}} \sim t(n-1)$$
</div>

## 가설

<div style="overflow: auto;">

> $$H_0 : \mu_{before}-\mu_{after}=\mu_D=0$$
</div>

## Example

~~~R
> shoes
$A
 [1] 13.2  8.2 10.9 14.3 10.7  6.6  9.5 10.8  8.8 13.3

$B
 [1] 14.0  8.8 11.2 14.2 11.8  6.4  9.8 11.3  9.3 13.6

> attach(shoes)
> t.test(A,B,paired=T) # paired=T - 대응표본

	Paired t-test

data:  A and B
t = -3.3489, df = 9, p-value = 0.008539
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -0.6869539 -0.1330461
sample estimates:
mean of the differences 
                  -0.41 

> D=A-B;D
 [1] -0.8 -0.6 -0.3  0.1 -1.1  0.2 -0.3 -0.5 -0.5 -0.3
> t.test(D,mu=0) # 일표본 T-검정으로도 가능

	One Sample t-test

data:  D
t = -3.3489, df = 9, p-value = 0.008539
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 -0.6869539 -0.1330461
sample estimates:
mean of x 
    -0.41 
~~~

+ $p=0.008539<0.05$ : 귀무가설 기각 - 차이가 있다 - B가 더 크다

***
# 일원배치 분산분석

> 세 집단 이상 평균비교(우단측 검정)

## 검정통계량
> 검정통계량 $$F=\frac{S_1^2}{S_2^2}=\frac{MSB}{MSW}$$

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

$$
총제곱합(Sum\ of\ Square\ Total)
=집단간\ 제곱합(Sum\ of\ Square\ Between)
+집단내\ 제곱합(Sum\ of\ Square\ Within)
$$
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

## Model(모형)

$$Y_{ij}=\mu_i+\epsilon_{ij}$$

+ 모집단의 형태

## Example

~~~R
> attach(Cars93)
> a1=lm(Width~AirBags,Cars93);a1

Call:
lm(formula = Width ~ AirBags, data = Cars93)

Coefficients:
       (Intercept)  AirBagsDriver only         AirBagsNone  
            71.875              -2.015              -4.287  
~~~

+ `lm()`은 Linear Model의 약자

~~~R
> anova(a1)
Analysis of Variance Table

Response: Width
          Df  Sum Sq Mean Sq F value    Pr(>F)    
AirBags    2  218.68 109.340  8.9856 0.0002767 ***
Residuals 90 1095.15  12.168                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
~~~

+ `AirBags` 집단간, `Residuals` 집단내
+ $p=0.0002767<0.05$ : 기각 - 차이가 있다
+ Pr(>F) - 우단측, $P1(F>F_0)$

~~~R
> a2=aov(Width~AirBags,Cars93);a2
Call:
   aov(formula = Width ~ AirBags, data = Cars93)

Terms:
                  AirBags Residuals
Sum of Squares   218.6799 1095.1481
Deg. of Freedom         2        90

Residual standard error: 3.488311
Estimated effects may be unbalanced
> TukeyHSD(a2)
  Tukey multiple comparisons of means
    95% family-wise confidence level

Fit: aov(formula = Width ~ AirBags, data = Cars93)

$AirBags
                                    diff       lwr        upr     p adj
Driver only-Driver & Passenger -2.014535 -4.448921  0.4198512 0.1250460
None-Driver & Passenger        -4.286765 -6.807012 -1.7665170 0.0003127
None-Driver only               -2.272230 -4.180014 -0.3644452 0.0153291
~~~

+ 다중비교(기각되면 필수)

~~~R
> aggregate(Width,by=list(AirBags),mean)
             Group.1        x
1 Driver & Passenger 71.87500
2        Driver only 69.86047
3               None 67.58824
~~~

+ 1번 그룹의 평균이 큼을 볼 수 있음
+ $D\And P=D>N$

~~~R
> a3=lm(Price~AirBags,Cars93);a3

Call:
lm(formula = Price ~ AirBags, data = Cars93)

Coefficients:
       (Intercept)  AirBagsDriver only         AirBagsNone  
            28.369              -7.145             -15.195  

> anova(a3)
Analysis of Variance Table

Response: Price
          Df Sum Sq Mean Sq F value    Pr(>F)    
AirBags    2   2747 1373.49  21.178 2.901e-08 ***
Residuals 90   5837   64.86                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
> a4=aov(Price~AirBags,Cars93);a4
Call:
   aov(formula = Price ~ AirBags, data = Cars93)

Terms:
                 AirBags Residuals
Sum of Squares  2746.984  5837.037
Deg. of Freedom        2        90

Residual standard error: 8.05332
Estimated effects may be unbalanced
~~~

+ 유의확률 Check
+ `lm()` - `anova()`

~~~R
> TukeyHSD(a4)
  Tukey multiple comparisons of means
    95% family-wise confidence level

Fit: aov(formula = Price ~ AirBags, data = Cars93)

$AirBags
                                     diff       lwr       upr     p adj
Driver only-Driver & Passenger  -7.145494 -12.76566 -1.525327 0.0088790
None-Driver & Passenger        -15.195221 -21.01361 -9.376828 0.0000000
None-Driver only                -8.049726 -12.45415 -3.645302 0.0001033
~~~

+ diff가 음수이면 뒤가 더 큼

~~~R
> aggregate(Price,by=list(AirBags),mean)
             Group.1        x
1 Driver & Passenger 28.36875
2        Driver only 21.22326
3               None 13.17353
~~~

+ $D\And P>D>N$
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

> $$F=\frac{MSR}{MSE}$$

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

## Example
~~~R
> attach(Cars93)
> plot(Price,Length)
> a5=lm(Price~Length,Cars93);a5

Call:
lm(formula = Price ~ Length, data = Cars93)

Coefficients:
(Intercept)       Length  
   -41.5246       0.3331  

> anova(a5)
Analysis of Variance Table

Response: Price
          Df Sum Sq Mean Sq F value    Pr(>F)    
Length     1 2177.3  2177.3  30.925 2.663e-07 ***
Residuals 91 6406.8    70.4                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
~~~

![plot()](https://user-images.githubusercontent.com/42334717/72406438-8cd02380-379f-11ea-95bb-20f680436dde.png)

+ Length - `SSR`
+ Residual - `SSE`
+ 기각 - 영향력 있다
~~~R
> summary(a5)

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

+ Coefficient : Esitimate - $\beta_0, \beta_1$

+ `lm()`
+ `anova()`
+ `summary()` 