---
title: CAM and CNC
date: 2020-09-02 13:19:07
categories:
- Etc.
tags:
- B.S. Course Work
- Mechanical Engineering
---
# Introduction

## 유연생산시스템(FMS)

> 생산성과 유연성의 양립을 목적으로 한 시스템으로서, 가공, 반송, 자재의 착탈, 제어의 기능을 유기적으로 결합한 자동화된 생산시스템

+ FMS(Flexible Manufacturing System) : 기계와 외부적인 연결
  + 다양한 순서의 자동 재료흐름
  + AGV(Automatic Guidied Vehicle), 산업용 Robot, Tooling, Pallet, Fixture
  + 공장무인화
+ FMC(Flexible Manufacturing Cell) : 독자적으로 작동
  + Machining center : 공작기계 내에서의 Pallet를 이용한 자동생산, 자동공구교환
  + 기계간 Pallet는 도움 필요
  + 40 ~ 800 부품
+ FTL(Flexible Transfer Line) : 기계와 내부적인 연결
  + 자동재료 이송시스템, NC 공작기계, 자동헤드 교환장치
  + 직접적인 재료흐름, 공작물의 순환운동
  + 1,500 ~ 15,000

|    특성\종류    | Transfer Line(FTL) |  FMS  | Standard-alone NC machines(FMC) |
| :-------------: | :----------------: | :---: | :-----------------------------: |
|     생산량      |         상         |  중   |               하                |
| 제품종류/유연성 |         하         |  중   |               상                |

<!-- More -->

+ FMS 구성요소
  + A group of workstations(CNC machine tools)
    + Machining centers
    + Milling modules
    + Turning modules
    + Assembly workstations
    + Inspection stations
    + Sheet metal processing machines
    + Forging stations
  + Automated material handling and storage systems
    + AGV(Automated Guided Vehicle)
    + Tool transporter
    + Industrial robots
    + Pallet and fixture
    + Conveyor
    + Stacker crane
  + Computer control systems
    + Control of each workstation(CAM)
    + Distribution of control instructions to workstation
    + Production control
    + Traffic control
    + Work handling system monitoring
    + Tool control
    + System performance monitoring and reporting
    + Production planning and management

## CAD / CAM

+ CIM : Computer Integrated Manufacturing, 통합생산시스템
  + 공장 자동화 기술
  + Database(EDB, MDB)
  + 통신기술
  + Web based
+ FMS : Flexible Manufacturing System, 유연생산시스템
  + 공장 자동화 기술 : CAD / CAM, CNC machining
+ CAD / CAM : Computer Aided Design and Computer Aided Manufacturing
  + CAD : Computer를 이용한 부품의 모델링
    + Wire frame : 제도용
    + Surface model : 금형가공용
    + Solid model : 해석용
  + CAM : 기계가공을 위한 모델링과 CNC machine을 작동시키기 위한 NC code 생성
    + Input : CAD
    + Output : NC code
  + NC code
    + NC 가공을 위한 표준화된 수치데이터 형식
    + Machining center의 Controller 명령문
  + Part program : 가공을 위한 일련의 NC code
+ CAD / CAM의 데이터 교환 : IGES, DXF, STEP
+ CAPP : Computer Aided Process Plan

***

# 절삭 가공

+ 절삭 가공이란?
  + 상대적으로 경도가 높은 날끝공구(Cutting Tool)를 사용하여 피가공물(Workpiece)의 불필요한 부분을 칩(Chip)의 형태로 제거하여 원하는 형태로 만드는 작업
+ 절삭 가공의 특징
  + 정밀 가공 가능
  + 가공에 따른 소재 내부의 물성 변화 적음
  + 다양한 형상가공(Flexible Process)
  + 칩의 발생에 따른 재료 손실
+ 절삭 가공을 수행하기 위한 3요소
  + 공작기계
  + 공구
  + 공작물

## 공작기계

+ 공작기계란?
  + 기계를 만드는 기계
  + 일반적으로는 절삭, 연삭 등과 같이 재료를 가공하여 원하는 형상으로 만들어 내는 기계
+ 공작기계의 분류
  + 비절삭 공작기계 : 주조, 소성가공, 용접 등과 같이 Chip을 발생하지 않고 가공
  + 절삭 공작기계 : 선삭, 밀링, 연삭 등 Chip을 발생시키면서 가공
  + 좁은 의미의 공작기계 : 절삭 공작기계를 의미

### 공작기계의 분류

+ 금속공작기계(Metal Cutting Machining Tool)
  + 범용 공작기계
    + 절삭공구 사용 기계
      + 고정공구 사용 기계
        + 선삭(Lathe)
        + 형삭(Shaper)
        + 평삭(Planer)
      + 회전공구 사용 기계
        + 밀링(Milling M/C)
        + 드릴링(Drilling M/C)
        + 보링(Boring M/C)
        + 쏘잉(Sawing M/C)
    + 연삭공구 사용 기계
      + 연삭(Grinding M/C)
      + 호닝(Honing M/C)
  + 전용 공작기계
    + 전용기(Special Purpose M/C)
  + NC 공작기계
    + NC Lathe
    + NC Drilling M/C
    + NC Milling M/C
    + NC Boring M/C
    + NC Grinding M/C
    + Machining Center
+ 금속가공기계(Metal Forming Machine Tool)
  + Press
  + Rolling M/C
  + Shearing M/C
  + Bending M/C

### NC 공작기계에 의한 가공의 특성

+ 높은 공작 정밀도(Accuracy)
  + 주축 회전정밀도
  + 안내면 직선 정밀도
  + 온도변화에 대한 변형
  + 진동
  + Etc.
+ 우수한 가공능률(Efficiency)
  + 절삭효율
    + 유효 절삭시간
    + 준비시간
    + 유휴시간
+ 융통성(Flexibility)
  + 프로그램에 의한 가공의 자동화
    + NC code
    + Controller
+ 안전성(Safety)
  + 작업자에 대한 안정성
  + 기계 자체의 안정성

### 공작기계의 운동

+ 공작기계의 가공 원칙
  + 절삭공구와 공작물간에 적절한 상대운동을 통하여 요구되는 형상 생성
+ 절삭운동과 이송운동 : 공작기계로부터 공급되는 상대운동
  + 절삭운동(Cutting motion, 주운동)
    + 기계가공 수행을 위한 총동력의 대부분을 사용
    + Chip의 길이 방향으로 공구가 움직이는 운동
  + 이송운동(Feed motion)
    + 가공물을 절삭 방향으로 피이드 하는 운동
    + 기계가공 수행을 위해 필요한 총동력의 소량을 사용

### 좌표계의 정의

+ Z축 운동
  + 주운동을 제공하는 기계의 주축에 평행하게 정렬
  + 주축이 없는 기계 : 공작물 지탱면에 수직으로 정렬
  + +Z 운동 : 공작물과 공구대 사이의 거리를 증가시키는 방향
+ X축 운동
  + 공작물 지탱면에 수평하고 평행
  + 주축이 없는 기계 : 주절삭 방향에 평행하고 주운동 방향이 플러스 방향
  + 공작물이 회전하는 기계 : 횡이송대에 방사형이고 평행
  + +X 운동 : 공구가 공작물의 회전축으로부터 멀어졌을 때의 공구 운동으로 정의
+ Y축 운동
  + 좌표계를 완성하는 방향

## 선삭

### 선반의 구성

+ 주축에 고정한 공작물을 회전, 공구대에 설치된 공구에 절삭깊이와 이송을 주어 공작물을 절삭
+ 베드 : 다른 구성요소들의 지지 역할
+ 왕복대(Carriage) : 베드의 안내면(Slide way)을 따라 이동
+ 주축대(Headstock) : 베드에 고정
+ 정밀도에 중요한 요소
  + 주축 흔들림(주축 베어링)
  + 이송운동의 정밀도(베드, Linear guide 정밀도)

![lathe](/images/cam-and-cnc/lathe.jpg)

### 선삭의 절삭운동(Cutting motion in lathe)

+ 주운동(Primary motion) : 공작기계의 주운동으로부터 야기되는 운동
+ 이송운동(Feed motion) : 공작기계 이송운동으로 야기되는 운동
+ 합 절삭 운동(Resultant motion) : 공구 주운동과 이송운동의 합

### 선삭공구 형상

+ $X_r$ : 주절삭날각(Major edge angle)
+ $a_c$ : 미변형 칩두께(Underformed chip thickness)
  + $a_c=fsin(X_r)$
+ $A_c$ : 한개의 철삭날에 의해 제거될 재료의 단면적(미절삭 칩 단면적, Cutting area)
  + $A_c=fa_{p1}$

<img src="/images/cam-and-cnc/lathe.png" alt="lathe" width="684" />

### 보링(Boring)

+ Drilling 또는 주조 등에서 이미 뚫린 구멍을 확대하거나 내부를 완성하는 가공
+ 선삭과 같음
+ 정밀도 증가

## Drilling

+ 다인공구인 Drill을 회전시키면서 축방향으로 이송을 주어 주로 구멍가공을 수행하는 공작기계를 Drilling machine이라 함
+ Drilling machine의 크기는 가공할 수 있는 구멍의 최대지름 및 길이 또는 Column 내측에서 주축까지의 최대거리와 주축 하단에서 Table 상면까지의 최대거리로 표시

### 드릴에서의 절삭(Cutting in drilling)

+ 미변형 칩두께(Undeformed chip thickness)
  + $a_c=\frac{f}{2}sin\chi_r$
  + $\chi_r$ : 주절삭날
+ 가공시간(Cutting time)
  + $t_m=\frac{l_w}{fn_t}$
+ 금속제거율(Material removal rate)
  + $Z_w=\frac{\pi}{4}d_m^2v_f=\frac{\pi fd_m^2n_t}{4}$

## Milling

### 밀링 머신의 구성과 분류

![milling](/images/cam-and-cnc/milling.png)

+ 테이블(Table)
+ Saddle
+ Knee
+ Overarm
+ 주축대(Head)

### 평면밀링에서의 기하학

+ 공구 1회전당 공작물 이동거리(Feed per rotation)
  + $f=\frac{v_f}{n_t}$
  + $v_f$ : 공작물 이송속도(Feed velocity)
  + $n_t$ : 절삭공구 회전속도(rpm)
+ 이송물림(Feed per tooth)
  + $a_f=\frac{f}{N}$
  + $N$ : 날수
+ 최대 미변형 칩두께(Max. undeformed chip thickness)
  + $a_{cmax}=\frac{v_fsin\theta}{Nn_t}$
  + $cos\theta=1-\frac{2a_e}{d_t}$
  + $d_t$ : 절삭공구 지름(Cutter diameter)
  + $a_e$ : 절삭깊이(Depth of cut)

### 정면밀링 구조

+ 회전당 이송량(Feed per revolution)
  + $f=\frac{v_f}{n_t}$
  + $v_f$ : 공작물 이송속도(Work feed)
  + $n_t$ : rpm of tool
+ 미변형 칩두께(Undeformed chip thickness)
  + $a_{cmax}=\frac{v_f}{Nn_t}$

## 연삭

> 연삭 숫돌 입자(Abrasive grain)의 절삭작용으로 가공물에서 미소 chip이 발생하도록하는 가공

+ 장점
  + 연삭 숫돌 입자의 경도가 높기 때문에 경질재료의 가공에 용이
  + 생성되는 chip이 매우 작아 높은 가공 정밀도

## NC machining center

### 자동화의 종류

+ 고정 자동화(Fixed automation)
  + 장비의 자동화 : 초기 투자비가 많이 듦
  + 유연성이 떨어짐
  + Transfer line, 자동선반(Automatic lathe), 전용장비
+ 프로그램 자동화
  + 프로그램 순차제어
    + Timer, Relay, Controller, Limit switch
    + 순차적인 자동화(PLC : Programmable Logic Controller)
  + 수치제어(Numerical Control)
    + NC controller에 의한 동시제어 가능
    + 수치에 의한 제어 가능
    + NC controller, AC, DC motor, Step motor에 의해서 작동, NC code에 의한 명령문 작성
    + CNC lathe, Machining center, Robot manipulator

### NC 공작기계의 구조

+ NC 공작기계의 구성
  + 명령 프로그램(NC code)
  + 제어기(Controller)
  + 공작기계
+ 명령 프로그램
  + NC code : 알파벳과 수치로 구성, 공작기계의 모든 동작을 지시
  + 공작기계와 무관한 Part programer에 의해서 작성
  + 공구의 고동을 지시 : 위치, 속도, 가속도(G code)
  + 기타 동작 지시 : 냉각제 공급, 자동공구교환(M-code)
+ NC controller
  + NC code를 받아들여서 공작기계의 다양한 행동을 제어하는 신호로 변환
  + Interpolation(보간기능 변환기)를 통하여 각 축의 모터구동을 위한 신호제작
+ NC 공작기계
  + 스핀들과 테이블의 자동구동장치
  + 다축동시제어, 자동공구교환기
  + NC lathe, Machining Center, NC drilling machine, Tapping, Boring

***

# NC Programming

> NC Programming을 Part Programming이라고도 함

+ NC Programming의 과정
  + 설계된 도면(Part drawing)의 판독
  + NC 가공을 위한 공정계획(Process plan) 작성
  + NC code를 이용한 파트프로그램 작성
  + NC 프로그램을 NC 기계에 입력 또는 Network를 통해 전송
+ NC 가공을 위한 공정계획
  + 제품도면에서 NC 가공부위를 선정
  + 해당 부위의 가공에 적합한 NC 기계, 공구(절삭방법), 고정구 등의 선정
  + 절삭가공 순서(출발점, 황삭/중삭/정삭계획 등) 결정
  + 실제 NC 공구(Cutter, Adapter, Holder 등) 선정 및 수배
  + 절삭조건(Spindle, Feed rate, Coolant 등) 결정

## 기본 NC 코드 구성

+ 시작과 끝 : `%`
+ 주석(Comment) : `()`
+ Word : `A~Z`(Address) + `수치`
+ Block : Word로 이루어짐

## NC Address

|        기능        | Address |                 비고                  |
| :----------------: | :-----: | :-----------------------------------: |
|   프로그램 번호    |    O    |             프로그램 번호             |
|       문번호       |    N    |             NC 블록 번호              |
|       좌표값       | X, Y, Z |                좌표값                 |
|       좌표값       | A, B, C |             회전축의 각도             |
|       좌표값       | I, J, K |          원호의 중심점 좌표           |
|       좌표값       |    R    |                반지름                 |
|      준비기능      |    G    |            동작 모드 선정             |
|      이송속도      |    F    |           이송속도(mm/min)            |
|   주축 회전 속도   |    S    |          주축 회전 속도(rpm)          |
|     공구 번호      |    T    |               공구 번호               |
|     보조 기능      |    M    | 기계 제어 지령(다양한 보조 기능 수행) |
| 옵셋 레지스터 번호 |  D, H   |          옵셋 레지스터 번호           |

## 좌표계

1. `Z축` : 주축 Spindle
2. `X축` : 수평(작업자의 좌우)
3. `Y축` : 오른손 법칙
+ `+` : 공구와 공작물이 멀어지는 방향

### 공작물 좌표계

+ 공작물이 회전하는 공작기계(선반)
  1. `Z축` : 공작물의 회전축
     + `+` : 주축이 공구를 보는 방향
  2. `X축` : 공구의 운동방향
     + `+` : 주축의 회전 중심 -> 밀어지는 방향
  3. `Y축` : `X축`, `Y축`이 직교
     + `+` : 오른손 좌표계
+ 공구가 회전하는 공작기계(Milling, Drilling)
  1. `Z축` : 주축(공구 회전축)
     + `+` : 공작물이 주축을 바라보는 방향
  2. `X축`
     + `Z축 수평` : 직교하는 수평축, `+ Y축`이 윗쪽이 되도록
     + `Z축 수직` : 기계 앞에 서서 오른쪽이 `+ X축`
+ 공작기계 좌표축
  + 공구를 이동 : 표준 좌표계와 동일(`Z축`)
  + 공작물을 이동 : 표준 좌표계와 반대방향(`X축`, `Y축`)

### 좌표값 워드

+ 최소설정단위(BLU, Blank Length Unit, 장비의 정밀도) 입력 방식
  + Ex) (x,y) = (50,23.4567), BLU = 0.001mm -> `X50000 Y23457`
+ 소수점 입력 방식
  + Ex) `X50. Y23.457`

### 공구번호 및 절삭조건의 지정(T, F, S)

+ `T12` : 12번에 있는 공구(공구 매거진 Tool slot 번호)
+ `F500` : 500mm/min(Feed rate, 이송속도)
+ `S1500` : 1500rpm(Spindle speed, 주축 회전 속도)
+ Ex) `X50. Y23.457 F200 S1000`

### 보조 기능(Miscellaneous function : M code)

> NC 프로그램을 제어하고 기계의 ON/OFF 제어기를 제어

|    구분     |  M code  |               기능                |
| :---------: | :------: | :-------------------------------: |
| 프로그램 끝 |   M00    |        프로그램 정지(Stop)        |
| 프로그램 끝 | M02, M30 |   프로그램 완료 및 재수행 준비    |
|  주축 회전  |   M03    | 시계방향으로 주축 회전(오른 공구) |
|  주축 회전  |   M04    |     반시계방향으로 주축 회전      |
|  주축 회전  |   M05    |          주축 회전 정지           |
|  공구 교환  |   M06    |          공구 교환 명령           |
|   절삭유    |   M08    |             절삭유 ON             |
|   절삭유    |   M09    |            절삭유 OFF             |

### 준비 기능(Preparatory function : G code)

|             구분             | G code |                  기능                   |
| :--------------------------: | :----: | :-------------------------------------: |
|        공구 이동 형태        |  G00   |          급속 이동(위치 제어)           |
|        공구 이동 형태        |  G01   |   직선 보간(주어진 속도로 직선 이동)    |
|        공구 이동 형태        |  G02   |              원호 보간 CW               |
|        공구 이동 형태        |  G03   |              원호 보간 CCW              |
|   공구 일시 정지(One-shot)   |  G04   |     지정된 시간만큼 공구 이동 정지      |
|          평면 설정           |  G17   | XY평면(2차원 밀링에서의 원호 보간 평면) |
|          평면 설정           |  G18   |  ZX평면(NC 선반에서의 원호 보간 평면)   |
|          평면 설정           |  G19   |                 YZ평면                  |
|       좌표값 입력 단위       |  G20   |                inch 입력                |
|       좌표값 입력 단위       |  G21   |                 mm 입력                 |
|        공구 반경 보정        |  G40   |             반경 보정 취소              |
|        공구 반경 보정        |  G41   |     공구 진행 방향의 왼쪽으로 보정      |
|        공구 반경 보정        |  G42   |    공구 진행 방향의 오른쪽으로 보정     |
|       좌표값 입력 형태       |  G90   |           좌표의 절댓값 입력            |
|       좌표값 입력 형태       |  G91   |           좌표의 증분값 입력            |
| 공작물 좌표계 설정(One-shot) |  G92   |           공작물 좌표계 설정            |

> One-shot : 한 그룹 내에서는 어느 한 값이 항상 선택됨. 한 번 선택되면 다른 값으로 변경 전까지 계속 유효

## 3차원 자동 NC 프로그램의 장점

+ 배우고 사용하기 쉬움
+ 프로그램 작성 시간이 짧음
+ 검증이 용이하고 오류가 적음
+ 효율적인 NC 가공이 가능(효율적 경로 및 절삭 조건)

***

# 자유곡면의 NC 절삭가공

> 자유곡면 : 한 수식으로 정의할 수 없는 곡면

## NC 가공에서의 고려사항

+ 황삭 계획 및 허용공차 지정(Roughing plan and allowance)
+ 가공경로 계획 및 영역가공(Tool path planning)
+ 직선보간길이 계산(Step length calculation)
+ 경로간 간격 계산(Path interval calculation)
+ 공구간섭 방지(Over-cut preotection)
+ 절삭조건 지정(Cutting condition)

## 곡면의 NC 가공을 위한 미분기하학

+ 곡면의 법선벡터와 CL 데이터 계산
  + $n$ : 접점 $r_c$에서의 단위법선벡터
  + $r_u=\frac{\partial r(u,v)}{\partial u}$ : $u$방향의 접선벡터
  + $r_v=\frac{\partial r(u,v)}{\partial v}$ : $v$방향의 접선벡터
  + $r_L=r_c+R(n-u)$ : 공구의 위치를 나타내는 좌표값(CL data)
+ Unit normal vector
  + $n=\frac{r_u\times r_v}{|r_u\times r_v|}$
+ 곡선의 곡률(Curvature)
  + $\vec r(t)=x(t)\vec{i}+y(t)\vec{j}+z(t)\vec{k}$
  + $\dot{\vec{r}}(t)=\frac{dr(t)}{dt}$(곡선의 접선벡터)
  + $\vec T=\frac{\dot{\vec r}}{|\dot{\vec r}|}$(단위 접선벡터)
  + $s(t)=\int^t_0|\dot{r}(t)|$(곡선의 길이)
  + $k=|\frac{dT}{ds}|$(곡률 : 단위접선벡터의 변화율)
  + $k=\frac{|\dot{\vec r}\times\ddot{\vec r}|}{\dot s^3}$
    + $\dot{\vec r}=\frac{d\vec r}{dt},\ \ddot{\vec r}=\frac{d^2\vec r}{dt^2}$
  + 곡률반경(Radius curvature) = $\frac{1}{k}$
+ 곡면의 곡률
  + $\vec u(t)=(u(t),v(t))$
  + 곡면 $r(u, v)$에 놓인 3차원 곡선 $\vec r(t)$
    + $\vec r=\frac{d\vec r}{dt}=\frac{\partial \vec r}{\partial u}\frac{\partial u}{\partial t}+\frac{\partial \vec r}{\partial v}\frac{\partial v}{\partial t}=\vec r_u\dot u+\vec r_v\dot v$
  + 곡선의 이송속도 $\dot s$
    + $\dot s^2=|\dot{\vec r}|^2=(\dot{\vec r}\cdot\dot{\vec r})=\dot{\vec r}^T\cdot \dot{\vec r}=\dot u^TA^T\cdot A\dot u=\dot u^TG\dot u$
    + $G=A^TA$

## 황삭계획 및 가공허용공차지정

+ 다각형 소재로부터 황삭 가공(From polygonal shape)
  + 적정 절삭깊이(Depth of cut)로 여러 차례 거쳐 황삭
  + 몰드 금형의 캐비티나 코아 등 황삭 가공 시 이용
  + Many cutting required to be removed
+ 주조 금형을 통한 황삭 가공(From casted shape)
  + 최종 형상과 비슷한 소재로부터 황삭 가공
  + 주조와 같은 공정을 이용하여 최종 형상과 비슷한 소재
  + From near shape, cutting process can be saved

### Round endmill에 의한 가공

+ Ball endmill
  + 절삭성 불량(Cutting is not good at the center)
+ Round endmill
  + 밑날이 없음(No end cutting edge)
  + 주로 R부 가공(Mainly cutting by R part)
  + 상향절삭이 보장(CL 데이터 산출)

## 가공계획(Cutter path planning) 및 영역가공

+ Parametric method
  + Iso parametric curve를 따라 가공($u=u_1$ or $v=v_1$)
  + 공구접촉점(CC point) 기준
  + 수치적 계산 간단(사각형 곡면 가공시 적합)
  + 보통 곡률이 큰 방향 가공
+ Cartesian method
  + 매개변수형 곡면시 수직평면으로 절단 후 평면 안에서 가공
  + CC-Cartesian : 공구의 접촉점(CC point)을 기준으로 가공
  + CL-Cartesian : 공구상의 기준점(CL point)을 기준으로 가공
  + 수치적 계산 복잡
  + 비 매개변수형 곡면시 절단 불필요
  + 불규칙한 형상에 적합

## 직선보간 길이 계산

$$
\delta_i\ : \ 내부공차
$$
$$
\delta_o\ : \ 외부공차
$$

+ CL Cartesian의 경우 원호보간 가능(G03)
+ L이 작으면 접촉점의 수 증가 -> 가공시간 증가

## 경로간 간격(Path interval)의 계산

+ $l_p\ :\ 경로간\ 간격$
+ $h\ :\ cusp높이$
+ $\rho\ :\ 곡면곡률반경$

<div style="overflow: auto;">

> $\rho\ 고려$
>> $$l=\frac{|\rho|[4(R+\rho)^2(h+\rho)^2-(\rho^2+2R\rho+(h+\rho)^2)^2]^{\frac{1}{2}}}{(R+\rho)(h+\rho)}$$
</div>

+ $\rho>0\ :\ 볼록곡면$
+ $\rho<0\ :\ 오목곡면$

> if $\rho \simeq \infty$
>> $$L=2\sqrt{h(2R-h)}$$

## 공구 간섭(Over cut) 방지

+ 오목한 곡면 부위의 곡률 반경이 공구 반경보다 작을 때 발생(Over cut)
+ 공구 간섭 방지 후 Under cut 발생
  + 작은 반경의 Ball end mill 가공 또는 방전가공, 사상가공
+ CL data : 곡면을 공구 반경만큼 Offset 시킨 곡면이 꼬이는 경우 간섭 발생
+ Under cut과 Over cut을 동시에 방지 : 사용 공구반경을 최소 곡률반경보다 작게(R < 1/Km)
+ 복합곡면의 경우 곡면이 만나는 부위에서 항상 공구 간섭 발생
  + CAD/CAM 구입 시 공구간섭현상의 처리능력 평가 필요