---
title: Crane and Transporter by Arduino
date: 2018-08-17 17:01:52
categories:
- Etc.
tags:
- Arduino
- C, C++
---

# 크레인 알고리즘
```C++
if(IR센서가 트랜스포터를 감지){
   모터로 줄을 늘임;
   전자석을 킴;
   모터로 줄을 당김;
   모터(바퀴)로 위치를 옮김;
   모터로 줄을 늘임;
   전자석을 끔;
   모터로 줄을 당김;
   모터(바퀴)로 원위치로 돌아가게 함;
}
else{
   모두 정지;
}
```
<!-- more -->

***
# IR센서
`VCC` : 전원
`GND` : 접지
`OUT` : 감지할 포트에 연결
***
# IR센서 코드
```Arduino
#define IR 13 //IR센서 포트 할당
#define LED_Red 10
#define LED_Green 11 //LED 포트 할당

void setup(){
pinMode(IR, INPUT);
pinMode(LED_Red, OUTPUT);
pinMode(LED_Green, OUTPUT);
Serial.begin(9600);
}

void loop(){
int IRval=digitalRead(IR);
if(IRval==1){
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, LOW);
  Serial.println(IRval);
  Serial.println("물체 감지 성공");
}
else{
  digitalWrite(LED_Red, LOW);
  digitalWrite(LED_Green, HIGH);
  Serial.println(IRval);
  Serial.println("물체 감지 실패");
}
delay(10);
}
```
***
# ON버튼과 OFF버튼, 그리고 IR센서를 통한 제어
```Arduino
#define S1 6
#define S2 7 //스위치 포트 할당
#define IR 13 //IR센서 포트 할당
#define LED_Yellow 9
#define LED_Red 10
#define LED_Green 11 //LED 포트 할당

void setup(){
pinMode(S1, INPUT_PULLUP);
pinMode(S2, INPUT_PULLUP);
pinMode(IR, INPUT);
pinMode(LED_Red, OUTPUT);
pinMode(LED_Green, OUTPUT);
pinMode(LED_Yellow, OUTPUT);
Serial.begin(9600);
}

int a=0;

void loop(){
int IRval=digitalRead(IR);
Serial.print("스위치 값 = ");
Serial.println(a);
if(digitalRead(S1)==1&&digitalRead(S2)==1){
  if(a>0){
  Serial.println("물체 감지 on");
  if(IRval==1){
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, LOW);
  digitalWrite(LED_Yellow, HIGH);
  Serial.println(IRval);
  Serial.println("물체 감지 성공");
}
else{
  digitalWrite(LED_Red, LOW);
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Yellow, HIGH);
  Serial.println(IRval);
  Serial.println("물체 감지 실패");
}
}
else if(a<0){
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Yellow, LOW);
  Serial.println("물체 감지 off");
}
else{
  Serial.println("초기 상태니 스위치를 누르십시오.");
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Yellow, HIGH);
}
}
else if(digitalRead(S1)==0){
  Serial.println("물체 감지 on");
  if(IRval==1){
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, LOW);
  digitalWrite(LED_Yellow, HIGH);
  Serial.println(IRval);
  Serial.println("물체 감지 성공");
}
else{
  digitalWrite(LED_Red, LOW);
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Yellow, HIGH);
  Serial.println(IRval);
  Serial.println("물체 감지 실패");
}
a=0;
a++;
}
else if(digitalRead(S2)==0){
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Yellow, LOW);
  Serial.println("물체 감지 off");
  a=0;
  a--;
}
delay(10);
}
```
***
# 라인트레이서 알고리즘(C)
```C++
#include<stdio.h>

int main(void) {
	printf("Sensor : white=0, black=1\nPressure : no pressure=0, pressure=1\nMotor : back=-1, stop=0, front=1\n\n");
	
	int S[4] = { 0, 0, 0, 0 };
	int P = 0;
	int M[2] = { 0, 0 };

	printf("센서부\nS0   S1\n\n\nS2   S3\n\n모터부\nM0   M1\n\n");
	int Button = 1;
	int i = 0;
	printf("초기화 스위치를 누르시려면 0, 트랜스포터의 상태를 그대로 두시려면 1을 입력해 주십시오. : ");
	scanf_s("%d", &Button);
	for (i = 0; i < 10000; i++) {//아두이노에서의 loop문
		printf("P의 INPUT을 입력하십시오 : ");
		scanf_s("%d", &P);
		if (Button == 0) {//초기화 버튼을 누른 상황
			printf("S[0]의 INPUT을 입력하십시오 : ");
			scanf_s("%d", &S[0]);
			printf("S[1]의 INPUT을 입력하십시오 : ");
			scanf_s("%d", &S[1]);
			printf("S[2]의 INPUT을 입력하십시오 : ");
			scanf_s("%d", &S[2]);
			printf("S[3]의 INPUT을 입력하십시오 : ");
			scanf_s("%d", &S[3]);
			if (S[0] == 0 && S[1] == 0 && S[2] == 0 && S[3] == 0) {//후진
				M[0] = -1;
				M[1] = -1;
			}
			else if (S[2] == 1 && S[3] == 0) {//3번 센서가 검은색을 인식
				M[0] = 0;
				M[1] = -1; //직진의 조건문을 만족하는 조건을 만들도록 가야함
			}
			else if (S[2] == 1 && S[3] == 1) {//정지
				M[0] = 0;
				M[1] = 0; //delay를 추가한다.(크레인을 올리거나 내려야함.)
				Button++;
			}
		}
		else if (Button == 1) {
			if (P == 1) {//압력이 존재할 때
				printf("S[0]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[0]);
				printf("S[1]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[1]);
				printf("S[2]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[2]);
				printf("S[3]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[3]);
				if (S[0] == 0 && S[1] == 0 && S[2] == 0 && S[3] == 0) {//직진
					M[0] = 1;
					M[1] = 1;
				}
				else if (S[0] == 0 && S[1] == 0 && S[2] == 1 && S[3] == 1) {//짐을 받고 잠시후 직진
					M[0] = 1;
					M[1] = 1; //delay를 추가한다.
				}
				else if (S[0] == 1 && S[1] == 0) {//1번 센서가 검은색을 인식
					M[0] = 0;
					M[1] = 1; //직진의 조건문을 만족하는 조건을 만들도록 가야함
				}
				else if (S[0] == 1 && S[1] == 1) {//정지
					M[0] = 0;
					M[1] = 0; //delay를 추가한다.(크레인을 올리거나 내려야함)
				}
			}
			else if (P == 0) {//압력이 존재하지 않을 때
				printf("S[0]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[0]);
				printf("S[1]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[1]);
				printf("S[2]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[2]);
				printf("S[3]의 INPUT을 입력하십시오 : ");
				scanf_s("%d", &S[3]);
				if (S[0] == 0 && S[1] == 0 && S[2] == 0 && S[3] == 0) {//후진
					M[0] = -1;
					M[1] = -1;
				}
				else if (S[0] == 1 && S[1] == 1 && S[2] == 0 && S[3] == 0) {
					M[0] = -1;
					M[1] = -1;
				}
				else if (S[2] == 1 && S[3] == 0) {//3번 센서가 검은색을 인식
					M[0] = 0;
					M[1] = -1; //직진의 조건문을 만족하는 조건을 만들도록 가야함
				}
				else if (S[2] == 1 && S[3] == 1) {//정지
					M[0] = 0;
					M[1] = 0; //delay를 추가한다.(크레인을 올리거나 내려야함.)
				}
			}
		}
		if (M[0] == 1 && M[1] == 1) {
			printf("전진\n\n");
		}
		else if (M[0] == 0 && M[1] == 0) {
			printf("정지\n\n");
		}
		else if (M[0] == 0 && M[1] == 1) {
			printf("좌회전\n\n");
		}
		else if (M[0] == -1 && M[1] == -1) {
			printf("후진\n\n");
		}
		else if (M[0] == 0 && M[1] == -1) {
			printf("후진 좌회전\n\n");
		}
	}
}
```
***
# 버튼을 이용한 LED제어
```Arduino
#define s1 6
#define s2 7
#define s3 8 //스위치 포트 할당
#define red 9
#define green 10
#define yellow 11
#define trred 3
#define trgreen 4 //LED 포트 할당

void setup(){
pinMode(s1, INPUT_PULLUP);
pinMode(s2, INPUT_PULLUP);
pinMode(s3, INPUT_PULLUP);
pinMode(red, OUTPUT);
pinMode(green, OUTPUT);
pinMode(yellow, OUTPUT);
pinMode(trred, OUTPUT);
pinMode(trgreen, OUTPUT);
Serial.begin(9600);
}

int a=0;
int b=0;
int c=0;
int d=0;

void loop(){
  if(digitalRead(s1)==1&&digitalRead(s2)==1&&digitalRead(s3)==1&&d==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    Serial.println("버튼을 아직 누르지 않았습니다.");
  }
  else if(digitalRead(s1)==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 0);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 0);
    a=0;
    b=0;
    c=0;
    d++;
    a++;
    Serial.println("초기화");
  }
  else if(digitalRead(s2)==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 0);
    digitalWrite(trgreen, 0);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    b++;
    Serial.println("시작");
  }
  else if(digitalRead(s3)==0){
    digitalWrite(red, 0);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    c++;
    Serial.println("정지");
  }
  else{
    if(a>0){
    digitalWrite(red, 1);
    digitalWrite(trred, 0);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 0);
    a=0;
    b=0;
    c=0;
    d++;
    a++;
    Serial.println("초기화");
    }
    else if(b>0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 0);
    digitalWrite(trgreen, 0);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    b++;
    Serial.println("시작");
    }
    else if(c>0){
    digitalWrite(red, 0);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    c++;
    Serial.println("정지");
    }
  }
  delay(10);
}
```
***
# 초기화버튼에 IR센서 추가
```Arduino
#define s1 6
#define s2 7
#define s3 8 //스위치 포트 할당
#define red 9
#define green 10
#define yellow 11
#define trred 3
#define trgreen 4 //LED 포트 할당
#define IR 13 //IR 포트 할당

void setup(){
pinMode(s1, INPUT_PULLUP);
pinMode(s2, INPUT_PULLUP);
pinMode(s3, INPUT_PULLUP);
pinMode(red, OUTPUT);
pinMode(green, OUTPUT);
pinMode(yellow, OUTPUT);
pinMode(trred, OUTPUT);
pinMode(trgreen, OUTPUT);
pinMode(IR, INPUT);
Serial.begin(9600);
}

int a=0;
int b=0;
int c=0;
int d=0;

void loop(){
  if(digitalRead(s1)==1&&digitalRead(s2)==1&&digitalRead(s3)==1&&d==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    Serial.println("버튼을 아직 누르지 않았습니다.");
  }
  else if(digitalRead(s1)==0){
    digitalWrite(yellow, 0);
    a=0;
    b=0;
    c=0;
    d++;
    a++;
    Serial.println("초기화");
    if(digitalRead(IR)==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 0);
    }
    else if(digitalRead(IR)==1){
    digitalWrite(red, 1);
    digitalWrite(trred, 0);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    }
  }
  else if(digitalRead(s2)==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 0);
    digitalWrite(trgreen, 0);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    b++;
    Serial.println("시작");
  }
  else if(digitalRead(s3)==0){
    digitalWrite(red, 0);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    c++;
    Serial.println("정지");
  }
  else{
    if(a>0){
    digitalWrite(yellow, 0);
    a=0;
    b=0;
    c=0;
    d++;
    a++;
    Serial.println("초기화");
    if(digitalRead(IR)==0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 0);
    }
    else if(digitalRead(IR)==1){
    digitalWrite(red, 1);
    digitalWrite(trred, 0);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    }
    }
    else if(b>0){
    digitalWrite(red, 1);
    digitalWrite(trred, 1);
    digitalWrite(green, 0);
    digitalWrite(trgreen, 0);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    b++;
    Serial.println("시작");
    }
    else if(c>0){
    digitalWrite(red, 0);
    digitalWrite(trred, 1);
    digitalWrite(green, 1);
    digitalWrite(trgreen, 1);
    digitalWrite(yellow, 1);
    a=0;
    b=0;
    c=0;
    d++;
    c++;
    Serial.println("정지");
    }
  }
  delay(10);
}
```
***
# 트랜스포터 보완
~~~Arduino
#define reset 22
#define start 23
#define halt 24//Button
#define pre A1//Pressure sensor
#define ir1 26
#define ir2 27
#define ir3 28
#define ir4 29//IR sensor
#define red 50
#define green 51
#define yellow 52//LED
#define m1f 30
#define m1b 31
#define m1p 2//Motor1
#define m2f 32
#define m2b 33
#define m2p 3//Motor2

void setup(){
  pinMode(reset, INPUT_PULLUP);
  pinMode(start, INPUT_PULLUP);
  pinMode(halt, INPUT_PULLUP);//Button
  pinMode(pre, INPUT);//Pressure sensor
  pinMode(ir1, INPUT);
  pinMode(ir2, INPUT);
  pinMode(ir3, INPUT);
  pinMode(ir4, INPUT);//IR sensor
  pinMode(red, OUTPUT);
  pinMode(green, OUTPUT);
  pinMode(yellow, OUTPUT);//LED
  pinMode(m1f, OUTPUT);//Front Motor 1
  pinMode(m1b, OUTPUT);//Back Motor 1
  pinMode(m2f, OUTPUT);//Front Motor 2
  pinMode(m2b, OUTPUT);//Back Motor 2
  Serial.begin(9600);
}

int a=0;
int b=0;
int c=0;

void loop(){
  Serial.print(analogRead(pre));
  if(digitalRead(reset)==1&&digitalRead(start)==1&&digitalRead(halt)==1){//아무것도 누르지 않음
    if(a==1){//초기화 누르고 이어진 상태
      a=1;
      b=0;
      c=0;
      Serial.println("초기화 가동");
      if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 후진 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 0);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, HIGH);
        analogWrite(m1p,100);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, HIGH);
        analogWrite(m2p,100);
        Serial.println("후진");
      }
      else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==0){//모터 후진 좌회전 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 0);
        digitalWrite(m1f, HIGH);
        digitalWrite(m1b, LOW);
        analogWrite(m1p,100);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, HIGH);
        analogWrite(m2p,100);
        Serial.println("후진 좌회전");
      }
      else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==1){//정지
        digitalWrite(red, 0);
        digitalWrite(green, 1);
        digitalWrite(yellow, 0);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, LOW);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, LOW);
        Serial.println("정지");
        delay(10000);//크레인 작동 시간
      }
    }
    else if(b==1){//시작 누르고 이어진 상태
      b=1;
      a=0;
      c=0;
     Serial.println("시작");
      if(analogRead(pre)>30){
        if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==1){//모터 전진 입력
          digitalWrite(red, 1);
          digitalWrite(green, 0);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, HIGH);
          digitalWrite(m1b, LOW);
          analogWrite(m1p,100);
          digitalWrite(m2f, HIGH);
          digitalWrite(m2b, LOW);
          analogWrite(m2p,100);
          Serial.println("전진");
       }
        else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 전진 입력
          digitalWrite(red, 1);
          digitalWrite(green, 0);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, HIGH);
          digitalWrite(m1b, LOW);
          analogWrite(m1p,100);
          digitalWrite(m2f, HIGH);
          digitalWrite(m2b, LOW);
          analogWrite(m2p,100);
          Serial.println("전진");
        }
        else if(digitalRead(ir1)==1&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 좌회전 입력
          digitalWrite(red, 1);
          digitalWrite(green, 0);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, LOW);
          digitalWrite(m1b, HIGH);
          analogWrite(m1p,100);
          digitalWrite(m2f, HIGH);
          digitalWrite(m2b, LOW);
          analogWrite(m2p,100);
          Serial.println("좌회전");
        }
        else if(digitalRead(ir1)==1&&digitalRead(ir2)==1&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//정지
          digitalWrite(red, 0);
          digitalWrite(green, 1);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, LOW);
          digitalWrite(m1b, LOW);
          digitalWrite(m2f, LOW);
          digitalWrite(m2b, LOW);
          Serial.println("정지");
          delay(10000);//크레인 작동 시간
        }
      }
      else if(analogRead(pre)<30){
        if(digitalRead(ir1)==1&&digitalRead(ir2)==1&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 후진 입력
          digitalWrite(red, 1);
          digitalWrite(green, 0);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, LOW);
          digitalWrite(m1b, HIGH);
          analogWrite(m1p,100);
          digitalWrite(m2f, LOW);
          digitalWrite(m2b, HIGH);
          analogWrite(m2p,100);
          Serial.println("후진");
        }
        else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 후진 입력
          digitalWrite(red, 1);
          digitalWrite(green, 0);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, LOW);
          digitalWrite(m1b, HIGH);
          analogWrite(m1p,100);
          digitalWrite(m2f, LOW);
          digitalWrite(m2b, HIGH);
          analogWrite(m2p,100);
          Serial.println("후진");
        }
        else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==0){//모터 후진 좌회전 입력
          digitalWrite(red, 1);
          digitalWrite(green, 0);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, HIGH);
          digitalWrite(m1b, LOW);
          analogWrite(m1p,100);
          digitalWrite(m2f, LOW);
          digitalWrite(m2b, HIGH);
          analogWrite(m2p,100);
          Serial.println("후진 좌회전");
        }
        else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==1){//정지
          digitalWrite(red, 0);
          digitalWrite(green, 1);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, LOW);
          digitalWrite(m1b, LOW);
          digitalWrite(m2f, LOW);
          digitalWrite(m2b, LOW);
          Serial.println("정지");
          delay(10000);//크레인 작동 시간 
        }
      }
      else{//정지
          digitalWrite(red, 0);
          digitalWrite(green, 1);
          digitalWrite(yellow, 1);
          digitalWrite(m1f, LOW);
          digitalWrite(m1b, LOW);
          digitalWrite(m2f, LOW);
          digitalWrite(m2b, LOW);
          Serial.println("정지?");
      }
    }
    else if(c==1){//정지 누르고 이어진 상태
      c=1;
      a=0;
      b=0;
      Serial.println("정지");
      digitalWrite(red, 0);
      digitalWrite(green, 1);
      digitalWrite(yellow, 1);
      digitalWrite(m1f, LOW);
      digitalWrite(m1b, LOW);
      digitalWrite(m2f, LOW);
      digitalWrite(m2b, LOW);
    }
    else{//초기 상태;
      digitalWrite(red, 1);
      digitalWrite(green, 1);
      digitalWrite(yellow, 1);
      Serial.println("초기값");
      digitalWrite(m1f, LOW);
      digitalWrite(m1b, LOW);
      digitalWrite(m2f, LOW);
      digitalWrite(m2b, LOW);
    }
  }
  else if(digitalRead(reset)==0){//초기화 버튼 누름
    a=1;
    b=0;
    c=0;
    Serial.println("초기화 가동");
    if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 후진 입력;
      digitalWrite(red, 1);
      digitalWrite(green, 0);
      digitalWrite(yellow, 0);
      digitalWrite(m1f, LOW);
      digitalWrite(m1b, HIGH);
      analogWrite(m1p,100);
      digitalWrite(m2f, LOW);
      digitalWrite(m2b, HIGH);
      analogWrite(m2p,100);
      Serial.println("후진");
    }
    else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==0){//모터 후진 좌회전 입력
      digitalWrite(red, 1);
      digitalWrite(green, 0);
      digitalWrite(yellow, 0);
      digitalWrite(m1f, HIGH);
      digitalWrite(m1b, LOW);
      analogWrite(m1p,100);
      digitalWrite(m2f, LOW);
      digitalWrite(m2b, HIGH);
      analogWrite(m2p,100);
      Serial.println("후진 좌회전");
    }
    else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==1){//정지
      digitalWrite(red, 0);
      digitalWrite(green, 1);
      digitalWrite(yellow, 0);
      digitalWrite(m1f, LOW);
      digitalWrite(m1b, LOW);
      digitalWrite(m2f, LOW);
      digitalWrite(m2b, LOW);
      Serial.println("정지");
      delay(10000);//크레인 작동 시간
    }
  }
  else if(digitalRead(start)==0){//시작 버튼 누름
    b=1;
    a=0;
    c=0;
    Serial.println("시작");
    if(analogRead(pre)>30){
      if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==1){//모터 전진 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, HIGH);
        digitalWrite(m1b, LOW);
        analogWrite(m1p,100);
        digitalWrite(m2f, HIGH);
        digitalWrite(m2b, LOW);
        analogWrite(m2p,100);
        Serial.println("전진");
      }
      else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 전진 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, HIGH);
        digitalWrite(m1b, LOW);
        analogWrite(m1p,100);
        digitalWrite(m2f, HIGH);
        digitalWrite(m2b, LOW);
        analogWrite(m2p,100);
        Serial.println("전진");
      }
      else if(digitalRead(ir1)==1&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 좌회전 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, HIGH);
        analogWrite(m1p,100);
        digitalWrite(m2f, HIGH);
        digitalWrite(m2b, LOW);
        analogWrite(m2p,100);
        Serial.println("좌회전");
      }
      else if(digitalRead(ir1)==1&&digitalRead(ir2)==1&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//정지
        digitalWrite(red, 0);
        digitalWrite(green, 1);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, LOW);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, LOW);
        Serial.println("정지");
        delay(10000);//크레인 작동 시간
      }
    }
    else if(analogRead(pre)>30){
      if(digitalRead(ir1)==1&&digitalRead(ir2)==1&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 후진 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, HIGH);
        analogWrite(m1p,100);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, HIGH);
        analogWrite(m2p,100);
        Serial.println("후진");
       }
      else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==0&&digitalRead(ir4)==0){//모터 후진 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, HIGH);
        analogWrite(m1p,100);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, HIGH);
        analogWrite(m2p,100);
        Serial.println("후진");
      }
      else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==0){//모터 후진 좌회전 입력
        digitalWrite(red, 1);
        digitalWrite(green, 0);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, HIGH);
        digitalWrite(m1b, LOW);
        analogWrite(m1p,100);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, HIGH);
        analogWrite(m2p,100);
        Serial.println("후진 좌회전");
      }
      else if(digitalRead(ir1)==0&&digitalRead(ir2)==0&&digitalRead(ir3)==1&&digitalRead(ir4)==1){//정지
        digitalWrite(red, 0);
        digitalWrite(green, 1);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, LOW);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, LOW);
        Serial.println("정지");
        delay(10000);//크레인 작동 시간
      }
    }
    else{//정지
        digitalWrite(red, 0);
        digitalWrite(green, 1);
        digitalWrite(yellow, 1);
        digitalWrite(m1f, LOW);
        digitalWrite(m1b, LOW);
        digitalWrite(m2f, LOW);
        digitalWrite(m2b, LOW);
        Serial.println("정지");
    }
  }
  else if(digitalRead(halt)==0){//정지 버튼 누름
    c=1;
    a=0;
    b=0;
    Serial.println("정지");
    digitalWrite(red, 0);
    digitalWrite(green, 1);
    digitalWrite(yellow, 1);
     digitalWrite(m1f, LOW);
     digitalWrite(m1b, LOW);
     digitalWrite(m2f, LOW);
     digitalWrite(m2b, LOW);
     Serial.println("정지");
  }
  delay(10);
}
~~~
