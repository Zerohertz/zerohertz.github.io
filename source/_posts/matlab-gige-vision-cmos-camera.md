---
title: Vision Data Acquisition MATLAB Source Code using GigE Vision CMOS Camera
date: 2022-01-04 23:39:26
categories:
- Etc.
tags:
- MATLAB
- Mechanical Engineering
---
# GigE Vision CMOS Camera

> GigE (Gigabit Ethernet) Vision 표준 기반 CMOS (Complementary metal-oxide-semiconductor) 카메라

## GigE Vision

> GigE (Gigabit Ethernet): 초당 기가비트 (Gbps)의 속도로 이더넷을 통해 데이터를 전송하는 기술

+ 1 Gbps = 125 MB/s (1 Gbit = 125 MB)

> GigE Vision: 고성능 산업용 카메라를 위한 인터페이스 표준

+ 이더넷 네트워크를 통해 고속 비디오 및 관련 제어 데이터를 전송하기 위한 framework 제공
+ Internet Protocol (IP) 표준을 기반으로 함

## CMOS

> CMOS (Complementary metal-oxide-semiconductor): microprocessor 혹은 SRAM 등의 디지털 회로를 구성하는 집적회로로 COS-MOS (Complementary-symmetry metal-oxide-semiconductor)로도 불림

+ 이미지 장치 분야에서는 CIS (CMOS Image Sensor)를 줄여 CMOS라 칭하는 경우가 있으며 기존의 기술인 CCD (Charge Coupled Device)를 대체
+ 빛에 의해 발생한 전자를 전압 형태로 변환해 전송하는 방식

<!-- More -->

***

# GigE Network Adapter Configuration in Mac OS

1. Ethernet을 통해 컴퓨터와 카메라 연결
2. 카메라에 할당된 IP 주소 및 서브넷 마스크를 수동으로 지정

![Setup IP](/images/matlab-gige-vision-cmos-camera/148081038-79787f47-7109-4ebf-8087-402f59aa1521.png)

3. `gigecamlist`로 연결 확인
4. `g = gigecam('IP Address')`를 통해 변수 `g`에 카메라 객체 할당

***

# Vision Data Acquisition

## Initial Setup

~~~Matlab
g.ExposureTime = 1000;
~~~

## Preview

~~~Matlab
preview(g)
closePreview(g)
~~~

## Capture and Save Vision Data

~~~Matlab
i = 1;

while true
    img = snapshot(g);
    imwrite(img, append(string(i),'.jpg'));
    pause(1) % Sampling Rate [fps]
    i = i + 1;
end
~~~