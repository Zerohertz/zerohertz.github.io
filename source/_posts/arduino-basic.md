---
title: Arduino Basic
date: 2018-08-17 17:01:00
categories:
- Etc.
tags:
- Arduino
- C, C++
---

# 변수 선언
```C++
int LED = 11;

int LED;
LED = 11;

#define LED 11 (메모리 사용 X) //define, const는 setup위에 종종 쓴다

const int LED = 11;
```
<!-- more -->
***
# 핀 모드 선언
```C++
void setup(){
pinMode(LED, OUTPUT);
pinMode(Sensor, INPUT);
Serial.Begin(9600); //아두이노 우노에서의 최대 속도
}
```
***
# OUTPUT입력
```C++
void loop(){
digitalWrite(LED, HIGH);
delay(1000); //단위 ms
digitalWrite(LED, LOW);
}
```
***
# 시리얼 모니터
```C++
void setup(){
Serial.begin(9600);
}

void loop(){
Serial.print("Hello, world\n"); //\n과 Serial.println(“”);는 엔터키와 같다
}
```
***
# loop내에서의 사칙연산
```C++
void loop(){
int a = 5;
int b = 3;
int c = 0;
c = a + b;
Serial.println(c); //“”필요 없음
}
```
***
# LED
긴 부분이 `+` 짧은 부분이 `-`
***
# 관계 연산자
`x == y` //x와 y가 같은가?
`x != y` //x와 y가 다른가?
`x > y` //x와 y보다 큰가?
`x < y` //x와 y보다 작은가?
`x >= y` //x와 y보다 크거나 같은가?
`x <= y` //x와 y보다 작거나 같은가?
***
# 논리 연산자
`x && y` : AND 연산, x와 y가 모두 참이면 참, 그렇지 않으면 거짓
`x || y` : OR연산, x나 y중에서 하나만 참이면 참, 모두 거짓이면 거짓
`!x` : NOT 연산, x가 참이면 거짓, x가 거짓이면 참
참 = 1, 거짓 = 0
***
# 조건 연산자
`max_value = (x > y) ? x : y;` //x > y가 참이면 x가 수식의 값이 된다.
//x > y가 거짓이면 y가 수식의 값이 된다.
***
# 지역 변수, 전역 변수, 자동 변수, 정적 변수
`지역 변수` : 함수 또는 블록 안에서 정의되는 변수. 지역 변수는 해당 블록이나 함수 안에서만 사용이 가능하다.
`전역 변수` : 함수의 외부에서 선언되는 변수. 전역 변수는 소스 파일의 어느 곳에서도 사용이 가능하다.
`자동 변수` : 지역 변수는 기본적으로 자동 할당된다. 변수를 선언한 위치에서 자동으로 만들어지고 블록을 벗어나게 되며 자동으로 소멸된다. 선언된 블록에서 사용이 끝나면 자동으로 메모리에서 제거되므로 메모리를 효율적으로 사용하게 된다.
`정적 변수` : 블록에서만 사용되지만 블록을 벗어나도 자동으로 제거되지 않는 변수.
***
# 조건문
조건에 따라 결정을 내리는 문장을 조건문이라고 한다.
***
> if문

`형식`
```C++
if(조건식)
     문장; //만약 조건식이 참인 경우에만 문장이 실행된다.
```
***
> if-else문

`형식`
 ```C++
if(조건식)
     문장1;
else
     문장2; //만약 조건식이 참이면 문장1이 실행된다. 그렇지 않으면 문장2가 실행된다.
```
***
> 다중 if문

`형식`
 ```C++
if(조건식1)
     문장1;
else if(조건식2)
     문장2;
else if(조건식3)
     문장3;
else
     문장4; //만약 조건식 1이 참이면 문장1이 실행된다. 그렇지 않고 조건식2가 참이면 문장2가 실행된다. 그렇지 않고 조건식3이 참이면 문장3이 실행된다. 그렇지 않으면 문장4가 실행된다.
```
***
> switch문

`형식`
 ```C++
switch(제어식)
{
   case c1:
     문장1;
     break; //제어식의 값이 c1이면 실행된다.
   case c2:
     문장2;
     break; //제어식의 값이 c2이면 실행된다.
   ...
    default:
     문장d;
     break; //일치하는 값이 없으면 실행된다.
}
```
***
> goto문

`형식`
```C++
goto error;
...
...
error:
     printf(“오류발생”); //조건없이 어떤 위치로 점프하게 만드는 문장이다. 하지만 프로그램을 복잡하게 만들기 때문에 사용을 장려하진 않음.
```
***
# 반복문
조건에 따라 반복을 하는 문장을 반복문이라고 한다.
***
> while문

`형식`
```C++
while(조건식)
     문장; //조건식이 참이면 문장을 반복 실행한다.
```
***
> do...while문

`형식`
```C++
do
     반복문장;
while(조건식); //일단 반복 문장을 실행한 후에 조건을 검사하여 반복 여부를 결정한다.
```
***
> for문

`형식`
```C++
 for(초기식; 조건식; 증감식)
     반복문장; //초기식을 실행한 후에 조건식의 값이 참인 동안, 반복 문장을 반복한다. 한번 반복이 끝날 때마다 증감식이 실행된다.
```
***
# 버튼인식
~~~Arduino
digitalRead(Button) == LOW //PULLUP일 때
~~~
***
# 버튼으로 LED제어
```Arduino
#define Button 9 //Button 포트 할당
#define LED_Red 10
#define LED_Green //LED 포트 할당

void setup(){
pinMode(Button, INPUT_PULLUP);
pinMode(LED_Red, OUTPUT);
pinMode(LED_Green, OUTPUT);
pinMode(LED_Yellow, OUTPUT);
Serial.begin(9600);
}
int a=0;
void loop(){
if(digitalRead(Button)==LOW){
  a++;
  Serial.println(a);
if(a<10){
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Red, LOW);
  delay(100);
  
  digitalWrite(LED_Red, HIGH);
  delay(100);
}
else{
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, LOW);
  delay(100);
  digitalWrite(LED_Green, HIGH);
  delay(100);
}
}
}
```
```Arduino
#define Button 9 //Button 포트 할당
#define LED_Red 10
#define LED_Green 11 //LED 포트 할당

void setup(){
pinMode(Button, INPUT_PULLUP);
pinMode(LED_Red, OUTPUT);
pinMode(LED_Green, OUTPUT);
pinMode(LED_Yellow, OUTPUT);
Serial.begin(9600);
}
int a=0;
void loop(){
if(digitalRead(Button)==LOW){
  a++;
  Serial.println(a);
if(a<5){
  digitalWrite(LED_Green, HIGH);
  digitalWrite(LED_Red, LOW);
  delay(100);
  digitalWrite(LED_Red, HIGH);
  delay(100);
}
else if(a>=5&&a<10){
  digitalWrite(LED_Red, HIGH);
  digitalWrite(LED_Green, LOW);
  delay(100);
  digitalWrite(LED_Green, HIGH);
  delay(100);
}
else
  a=0;
}
else{
  digitalWrite(LED_Red, 1);
  digitalWrite(LED_Green, 1);
  a=0;
}
}
```
