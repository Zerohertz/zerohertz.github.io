---
title: C++ Basic (4)
date: 2019-03-01 20:21:50
categories:
- Etc.
tags:
- C, C++
---
# 생성자와 소멸자

+ 객체를 초기화(멤버 변수의 값을 초기화)
+ 생성자는 반드시 public으로 선언되어야 한다

> 디폴트 생성자

<!-- more -->

~~~C++
#include <iostream>
#include <string>
#include <windows.h>
using namespace std;

class car {
private:
	int speed;
	int gear;
	string color;
public:
	car()
	{
		cout << "디폴트 생성자 호출" << endl;
		speed = 0;
		gear = 1;
		color = "white";
	}
};

int main() {
	car mycar;

	system("pause");
}
~~~
> 실행결과

~~~C++
디폴트 생성자 호출
계속하려면 아무 키나 누르십시오 . . .
~~~
> 클래스 외부 정의

~~~C++
car::car()
     	{
     		cout << "디폴트 생성자 호출" << endl;
     		speed = 0;
     		gear = 1;
     		color = "white";
     	}
~~~
