---
title: R Mid-term
date: 2020-01-06 01:06:20
categories:
- Etc.
tags:
- R
- Statistics
- B.S. Course Work
---
# setwd(), getwd()

~~~R
setwd("C:/Users/SW05/Downloads")
getwd()
~~~
<!-- more -->
***
# summary(), str()

~~~R
> summary(air)
     Ozone           Solar.R           Wind             Temp           Month            Day      
 Min.   :  1.00   Min.   :  7.0   Min.   : 1.700   Min.   :56.00   Min.   :5.000   Min.   : 1.0  
 1st Qu.: 18.00   1st Qu.:115.8   1st Qu.: 7.400   1st Qu.:72.00   1st Qu.:6.000   1st Qu.: 8.0  
 Median : 31.50   Median :205.0   Median : 9.700   Median :79.00   Median :7.000   Median :16.0  
 Mean   : 42.13   Mean   :185.9   Mean   : 9.958   Mean   :77.88   Mean   :6.993   Mean   :15.8  
 3rd Qu.: 63.25   3rd Qu.:258.8   3rd Qu.:11.500   3rd Qu.:85.00   3rd Qu.:8.000   3rd Qu.:23.0  
 Max.   :168.00   Max.   :334.0   Max.   :20.700   Max.   :97.00   Max.   :9.000   Max.   :31.0  
 NA's   :37       NA's   :7          
> str(air)
'data.frame':	153 obs. of  6 variables:
 $ Ozone  : int  41 36 12 18 NA 28 23 19 8 NA ...
 $ Solar.R: int  190 118 149 313 NA NA 299 99 19 194 ...
 $ Wind   : num  7.4 8 12.6 11.5 14.3 14.9 8.6 13.8 20.1 8.6 ...
 $ Temp   : int  67 72 74 62 56 66 65 59 61 69 ...
 $ Month  : int  5 5 5 5 5 5 5 5 5 5 ...
 $ Day    : int  1 2 3 4 5 6 7 8 9 10 ...
~~~
***
# Data 취사 선택
> indexing - []

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
# sort(), order()

~~~R
> Temp
  [1] 67 72 74 62 56 66 65 59 61 69 74 69 66 68 58 64 66 57 68 62 59 73 61 61 57 58 57 67 81 79 76 78
 [33] 74 67 84 85 79 82 87 90 87 93 92 82 80 79 77 72 65 73 76 77 76 76 76 75 78 73 80 77 83 84 85 81
 [65] 84 83 83 88 92 92 89 82 73 81 91 80 81 82 84 87 85 74 81 82 86 85 82 86 88 86 83 81 81 81 82 86
 [97] 85 87 89 90 90 92 86 86 82 80 79 77 79 76 78 78 77 72 75 79 81 86 88 97 94 96 94 91 92 93 93 87
[129] 84 80 78 75 73 81 76 77 71 71 78 67 76 68 82 64 71 81 69 63 70 77 75 76 68
> sort(Temp)
  [1] 56 57 57 57 58 58 59 59 61 61 61 62 62 63 64 64 65 65 66 66 66 67 67 67 67 68 68 68 68 69 69 69
 [33] 70 71 71 71 72 72 72 73 73 73 73 73 74 74 74 74 75 75 75 75 76 76 76 76 76 76 76 76 76 77 77 77
 [65] 77 77 77 77 78 78 78 78 78 78 79 79 79 79 79 79 80 80 80 80 80 81 81 81 81 81 81 81 81 81 81 81
 [97] 82 82 82 82 82 82 82 82 82 83 83 83 83 84 84 84 84 84 85 85 85 85 85 86 86 86 86 86 86 86 87 87
[129] 87 87 87 88 88 88 89 89 90 90 90 91 91 92 92 92 92 92 93 93 93 94 94 96 97
> sort(Temp,decreasing=T)
  [1] 97 96 94 94 93 93 93 92 92 92 92 92 91 91 90 90 90 89 89 88 88 88 87 87 87 87 87 86 86 86 86 86
 [33] 86 86 85 85 85 85 85 84 84 84 84 84 83 83 83 83 82 82 82 82 82 82 82 82 82 81 81 81 81 81 81 81
 [65] 81 81 81 81 80 80 80 80 80 79 79 79 79 79 79 78 78 78 78 78 78 77 77 77 77 77 77 77 76 76 76 76
 [97] 76 76 76 76 76 75 75 75 75 74 74 74 74 73 73 73 73 73 72 72 72 71 71 71 70 69 69 69 68 68 68 68
[129] 67 67 67 67 66 66 66 65 65 64 64 63 62 62 61 61 61 59 59 58 58 57 57 57 56
> order(Temp)
  [1]   5  18  25  27  15  26   8  21   9  23  24   4  20 148  16 144   7  49   6  13  17   1  28  34
 [25] 140  14  19 142 153  10  12 147 149 137 138 145   2  48 114  22  50  58  73 133   3  11  33  82
 [49]  56 115 132 151  31  51  53  54  55 110 135 141 152  47  52  60 108 113 136 150  32  57 111 112
 [73] 131 139  30  37  46 107 109 116  45  59  76 106 130  29  64  74  77  83  92  93  94 117 134 146
 [97]  38  44  72  78  84  87  95 105 143  61  66  67  91  35  62  65  79 129  36  63  81  86  97  85
[121]  88  90  96 103 104 118  39  41  80  98 128  68  89 119  71  99  40 100 101  75 124  43  69  70
[145] 102 125  42 126 127 121 123 122 120
> airquality[order(Temp),]
    Ozone Solar.R Wind Temp Month Day
5      NA      NA 14.3   56     5   5
18      6      78 18.4   57     5  18
25     NA      66 16.6   57     5  25
...
123    85     188  6.3   94     8  31
122    84     237  6.3   96     8  30
120    76     203  9.7   97     8  28
~~~

+ indexing 안에 `order()`
***
# read.table(), read.csv()

+ `.txt`를 열어서 Data 확인

~~~R
text1=read.table('Data.txt',header=T,na.strings='.')
text2=read.csv('Data.txt',header=T,na.strings='.')
~~~

+ 결측값 처리 - `na.strings='.'`
+ 문자형 변수 `factor`로 읽게 - `header=T`
***
# factor()

+ 성별이 1, 2로 되어 있으면 factor로 변환

~~~R
> a=read.table('women.txt')
> a
   qwer asdf gender
1    58  115      1
2    59  117      1
3    60  120      2
4    61  123      2
5    62  126      1
6    63  129      1
7    64  132      2
8    65  135      2
9    66  139      2
10   67  142      2
11   68  146      1
12   69  150      1
13   70  154      2
14   71  159      1
15   72  164      1
> str(a)
'data.frame':	15 obs. of  3 variables:
 $ qwer  : int  58 59 60 61 62 63 64 65 66 67 ...
 $ asdf  : int  115 117 120 123 126 129 132 135 139 142 ...
 $ gender: int  1 1 2 2 1 1 2 2 2 2 ...
> a$gender=factor(a$gender,labels=c('m','f'))
> a
   qwer asdf gender
1    58  115      m
2    59  117      m
3    60  120      f
4    61  123      f
5    62  126      m
6    63  129      m
7    64  132      f
8    65  135      f
9    66  139      f
10   67  142      f
11   68  146      m
12   69  150      m
13   70  154      f
14   71  159      m
15   72  164      m
> str(a)
'data.frame':	15 obs. of  3 variables:
 $ qwer  : int  58 59 60 61 62 63 64 65 66 67 ...
 $ asdf  : int  115 117 120 123 126 129 132 135 139 142 ...
 $ gender: Factor w/ 2 levels "m","f": 1 1 2 2 1 1 2 2 2 2 ...
~~~

+ `labels` 활용

~~~R
> a=read.table('women.txt')
> a
   height weight gender
1      58    115      1
2      59    117      2
3      60    120      2
4      61    123      1
5      62    126      2
6      63    129      2
7      64    132      1
8      65    135      2
9      66    139      2
10     67    142      2
11     68    146      1
12     69    150      2
13     70    154      2
14     71    159      2
15     72    164      2
> str(a)
'data.frame':	15 obs. of  3 variables:
 $ height: int  58 59 60 61 62 63 64 65 66 67 ...
 $ weight: int  115 117 120 123 126 129 132 135 139 142 ...
 $ gender: int  1 2 2 1 2 2 1 2 2 2 ...
> a$gender[which(a$gender==1)]='m'
> a$gender[which(a$gender==2)]='f'
> a
   height weight gender
1      58    115      m
2      59    117      f
3      60    120      f
4      61    123      m
5      62    126      f
6      63    129      f
7      64    132      m
8      65    135      f
9      66    139      f
10     67    142      f
11     68    146      m
12     69    150      f
13     70    154      f
14     71    159      f
15     72    164      f
> b=a
> b
   height weight gender
1      58    115      m
2      59    117      f
3      60    120      f
4      61    123      m
5      62    126      f
6      63    129      f
7      64    132      m
8      65    135      f
9      66    139      f
10     67    142      f
11     68    146      m
12     69    150      f
13     70    154      f
14     71    159      f
15     72    164      f
> a$gender=factor(a$gender,order=T,level=c('m','f'))
> b$gender=factor(b$gender,order=F,level=c('m','f'))
> a
   height weight gender
1      58    115      m
2      59    117      f
3      60    120      f
4      61    123      m
5      62    126      f
6      63    129      f
7      64    132      m
8      65    135      f
9      66    139      f
10     67    142      f
11     68    146      m
12     69    150      f
13     70    154      f
14     71    159      f
15     72    164      f
> b
   height weight gender
1      58    115      m
2      59    117      f
3      60    120      f
4      61    123      m
5      62    126      f
6      63    129      f
7      64    132      m
8      65    135      f
9      66    139      f
10     67    142      f
11     68    146      m
12     69    150      f
13     70    154      f
14     71    159      f
15     72    164      f
> str(a)
'data.frame':	15 obs. of  3 variables:
 $ height: int  58 59 60 61 62 63 64 65 66 67 ...
 $ weight: int  115 117 120 123 126 129 132 135 139 142 ...
 $ gender: Ord.factor w/ 2 levels "m"<"f": 1 2 2 1 2 2 1 2 2 2 ...
> str(b)
'data.frame':	15 obs. of  3 variables:
 $ height: int  58 59 60 61 62 63 64 65 66 67 ...
 $ weight: int  115 117 120 123 126 129 132 135 139 142 ...
 $ gender: Factor w/ 2 levels "m","f": 1 2 2 1 2 2 1 2 2 2 ...
~~~
***
# Data handling - rbind(case 추가) / cbind(변수 추가)

+ 하나는 .txt 하나는 .csv 어떻게 붙일지 판단하고 Data 병합

~~~R
> data1=read.table('women.txt',header=T)
> data2=read.csv('women.csv',header=T)
> data1
   qwer asdf gender
1    58  115      m
2    59  117      f
3    60  120      f
4    61  123      f
5    62  126      f
6    63  129      f
7    64  132      m
8    65  135      m
9    66  139      m
10   67  142      m
11   68  146      m
12   69  150      m
13   70  154      f
14   71  159      f
15   72  164      f
> data2
   height weight
1      58    115
2      59    117
3      60    120
4      61    123
5      62    126
6      63    129
7      64    132
8      65    135
9      66    139
10     67    142
11     68    146
12     69    150
13     70    154
14     71    159
15     72    164
> data=cbind(data1,data2)
> data
   qwer asdf gender height weight
1    58  115      m     58    115
2    59  117      f     59    117
3    60  120      f     60    120
4    61  123      f     61    123
5    62  126      f     62    126
6    63  129      f     63    129
7    64  132      m     64    132
8    65  135      m     65    135
9    66  139      m     66    139
10   67  142      m     67    142
11   68  146      m     68    146
12   69  150      m     69    150
13   70  154      f     70    154
14   71  159      f     71    159
15   72  164      f     72    164
~~~
***
# 변수계산

+ exp(), log(), sqrt(), 사칙연산
***
# head(), tail()

~~~R
> head(airquality)
  Ozone Solar.R Wind Temp Month Day
1    41     190  7.4   67     5   1
2    36     118  8.0   72     5   2
3    12     149 12.6   74     5   3
4    18     313 11.5   62     5   4
5    NA      NA 14.3   56     5   5
6    28      NA 14.9   66     5   6
> tail(airquality)
    Ozone Solar.R Wind Temp Month Day
148    14      20 16.6   63     9  25
149    30     193  6.9   70     9  26
150    NA     145 13.2   77     9  27
151    14     191 14.3   75     9  28
152    18     131  8.0   76     9  29
153    20     223 11.5   68     9  30
~~~
***
# 코딩변경 - 논리연산, cut()

~~~R
factor(name,labels=c(...))
cut(name,breaks=c(...),include.lowest=T/F,labels=c(...))
~~~
~~~R
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
# 제어문 - for, while, if~else if, ifelse(결과 단 2개)

> for

~~~R
for(i in num){
    ...
}
~~~

> while

~~~R
while(i<=num){
    ...
    i=i+1
}
~~~

> if

~~~R
if(case1){
    ...
}else if(case2){
    ...
}else{
    ...
}
~~~

> if else

~~~R
if(x<y) x else y
ifelse(x<y,x,y)
~~~
***
# 사용자정의함수, return()

~~~R
name=function(par1,par2,...){
    ...
    return(...)
}
~~~
~~~R
skew=function(x){
	sk=sum(((x-mean(x))^3)/sd(x)^3)*(1/(length(x)-1))
	return(sk)
}
kur=function(x){
	ku=sum(((x-mean(x))^4)/sd(x)^4)*(1/(length(x)-1))
	return(ku)
}
~~~
***
# 기술통계학

+ 자료 요약 및 정리 : 표(도수분포표)와 그래프()

+ 범주형 - 도수분포표(빈도를 그래프로 할 수 있긴 함)

+ 수치형 - 기술통계량(분포의 특성)
    + 대표값(중심위치)
        + 산술평균
        + 중위수
        + 최빈수(보통 범주형에서 사용, 수치형에선 이산형에서 가끔)
    + 산포도(흩어진 정도)
        + 표준편차(기존단위가 같음)
        + 분산(단위^2)
        + 변이(변동)계수(CV) : `sd()/mean()`
        + 사분위수범위(Q3-Q1) : `boxplot()`, `IQR()`
    + 비대칭도
        + 왜도

+ 오류데이터 찾기
    + 최소값
    + 최대값
    + 도수분포표
    + `summary()`

+ 관련된 변수 2개
    + 수치형 2개 : `cor(name1,name2)`
    + 범주형 2개 : 교차표(분할표)

+ 외부파일 읽기(`.txt`) - `read.table()` / `read.csv()` - `str()` - `summary()`
    + Data 여러개 - Data handling
        + 병합
            + `rbind` - 행(Case) 추가
            + `cbind` - 열(변수) 추가
        + 새로운 변수 생성
            + 변수계산 : 기존 변수를 가지고 계산을 통해 새로운 변수 생성
            + 코딩변경 : 기존 변수를 가지고 새로운 변수 생성(Ex. 학점 예제) - 보통 범주형
        + 데이터 취사선택 : `indexing - []`, `subset()`
        + 정렬 : `sort()`, `order()`

+ Slicing : `substr()`

+ NA : `mean(name,na.rm=T)`

+ 출력 : `cat()`, `sink()`, `pdf()`, `dev.off()`

+ 연산자, 제어문(반복문, 조건문), 함수

> 단위계산은 `*`, `/`만 계산

> 변동계수
>+ 평균의 차가 많은 집단끼리의 산포도 비교시 사용
>+ 단위가 다른 변수에 대한 산포도 비교 - 무차원수
>+ 극심한 비대칭일때 사용
***
# Etc.

+ 한 사람이나 한 개체의 데이터는 한 행으로 표시해야 한다
+ 범주형, 문자형 : factor
+ 수치형 : 정수(int), 실수(num)
+ 함수 : 내장 함수(R에 내장된 대부분의 함수명은 소문자), 외장 함수(사용자정의함수)
+ boxplot(), hist()
+ attach()
+ 조건 : 산술연산자, 비교연산자, 논리연산자 - ^ 제일 우선 () 활용으로 우선 순위 만들기
+ 시험지 안에 정보 다 있음 / read.table()과 비교
+ cbind() == data.frame() 상관 없음
+ 새로운 변수 생성 : 변수계산, 코딩변경
+ return() 2개는 전역변수 or c()
+ 왜도, 첨도 : 3, 4 승 n-1로 함수
+ switch() - X
+ sample(), substr(), IQR(), abs(), signif(), dim(), ncol(), nrow(), seq(), rep(), sink(), pdf()~dev.off()
+ 프 / 결 - 프로그램 / 결과 : 결과가 없을 수도 있음 - 수기로 작성, 다 프로그램
+ 기말은 서술도 나옴