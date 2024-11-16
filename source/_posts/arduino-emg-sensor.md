---
title: EMG Sensor
date: 2019-05-24 15:58:34
categories:
- Etc.
tags:
- Arduino
- C, C++
- B.S. Course Work
---
# EMG Sensor란?

![emg-sensor](/images/arduino-emg-sensor/emg-sensor.png)

> 근육의 수축 이완을 전압으로 나타냄
+ 노이즈가 많음
+ 매우 신화가 작아 증폭효과를 가지고 있음(계기증폭기)
+ 신호를 반전시켜줘야함
+ Signal processing(Low-pass filter)
+ 신호가 매우 작으므로 gel type의 전극을 붙임

<!-- more -->

![circuit-diagram](/images/arduino-emg-sensor/circuit-diagram.png)
![arduino](/images/arduino-emg-sensor/arduino.png)
***
# 오실로스코프로 EMG Sensor값 읽기

![experiment](/images/arduino-emg-sensor/experiment.jpeg)
***
# 아두이노로 EMG Sensor값 읽어보기

+ 0~1023(5V) -> `(val*5/1023-offset)*1000`(mV변환)

~~~C++
void setup() {
  Serial.begin(9600);
}

void loop() {
  float val = analogRead(A0);
  val = (val*5/1023)*1000-2420;
  Serial.println(val);
  delay(100);
}
~~~
***
# 신호 반전

~~~C++
void setup() {
  Serial.begin(9600);
}

void loop() {
  float val = analogRead(A0);
  val = (val*5/1023)*1000-2420;
  val = abs(val);
  Serial.println(val);
  delay(100);
}
~~~
![results](/images/arduino-emg-sensor/results.jpeg)
***
# Serial 통신

~~~C++
#include "stdio.h"
#include "Modules/SerialComm.h"

void main()
{
	CSerialComm SerialComm;

	if (SerialComm.connect("COM9"))
		printf("Serial port is connected!!\n");
	else
		printf("Sercial connection fail!!\n");

	while (1)
	{
		char data[64];
		SerialComm.readCommand(data);
		int i = 0;
		while (1)
		{
			printf("%c", data[i]);
			if (data[i] == '\n')
				break;
			i++;
		}
	}
}
~~~
***
# Low-pass filter

> Moving Average function 사용

~~~C++
#define WinSize 100
float data[WinSize];
int i = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int a = 0;
  float sol = 0;
  float val = analogRead(A0);
  val = (val*5/1023)*1000-2420;
  val = abs(val);
  a = i % WinSize;
  data[a] = val;
  if(i > WinSize ){
    for(int j = 0; j < WinSize ; j++){
      sol = sol + data[j] / WinSize;
    }
  }
  Serial.println(sol);
  delay(10);
  i++;
}
~~~

![graph-1](/images/arduino-emg-sensor/graph-1.png)
![graph-2](/images/arduino-emg-sensor/graph-2.png)