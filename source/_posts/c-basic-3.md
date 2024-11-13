---
title: C++ Basic (3)
date: 2019-01-07 20:23:44
categories:
- Etc.
tags:
- C, C++
---
# 형변환

+ (새로운 타입)수식
+ 새로운 타입(수식)

~~~C++
double d = 3.14;
int i;
i = (int)d;
i = int(d); // 새로운 형변환 형식
~~~
***
# 구조체

~~~C++
struct _point{
    int x;
    int y;
};
struct _point p1; // C 언어 방식
_point p2; // C++ 언어 방식
~~~
<!-- more -->

***
# 레퍼런스

> 변수에 별명을 붙여서 접근하는 것

~~~C++
int &ref = var;
~~~
+ `형 &`으로 선언 해준다
+ 포인터는 변수이지만 레퍼런스는 자신만의 메모리 공간이 할당되지 않음

> 오류나는 유형

~~~C++
int n = 10, m = 20;
int &ref = n;
ref = m; // 오류
~~~
+ 레퍼런스가 가리키는 대상은 변경될 수 없다

~~~C++
int &ref; // 오류
~~~
+ 선언과 동시에 초기화 시켜야 한다

~~~C++
int &ref = 10; // 오류
~~~
+ 상수로 초기화 시키면 오류가 난다

> 레퍼런스를 이용한 참조에 의한 호출

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

void swap(int &rx, int &ry);

int main() {
	int a = 100, b = 200;
	cout << "swap() 호출전: a = " << a << ", b = " << b << endl;
	swap(a, b);
	cout << "swap() 호출후: a = " << a << ", b = " << b << endl;

	system("pause");
}

void swap(int &rx, int &ry) {
	int tmp;
	tmp = rx; rx = ry; ry = tmp;
}
~~~
> 실행결과

~~~C++
swap() 호출전: a = 100, b = 200
swap() 호출후: a = 200, b = 100
계속하려면 아무 키나 누르십시오 . . .
~~~
***
# name space

~~~C++
namespace 이름{
    변수 정의;
    함수 정의
    클래스 정의;
    ...
}
~~~
~~~C++
using 이름공간::식별자; // 특정 이름 공간에서의 특정 식별자에 대해 이름 공간을 생략하여 접근 가능
using namespace 이름공간; // 이름 공간에 정의된 모든 식별자에 대해 이름 공간을 생략하여 접근 가능
~~~
***
# Class

+ `string`이 가장 기본적인 클래스
+ `field`에서 멤버 변수 선언
+ `method`에서 멤버 함수 선언

~~~C++
class car{
public:
    int speed;
    int gear;
    string color;
    
    void speedup(){
        speed += 10;
    }
    
    void speeddown(){
        speed -= 10;
    }
};

car globalcar; // 전역 객체 생성(인스턴스)
int main(){
    car localcar; // 지역 객체 생성(인스턴스)
}
~~~
+ `public`은 접근 지정자로서 클래스 외부에서 멤버를 자유롭게 사용할 수 있음을 의미한다
+ `class`에 값을 대입하면 안되고 `instance`에만 값을 대입해야한다

> 전용 멤버(private member) vs 공용 멤버(public member)

+ 각각 `private:`, `public:`과 같이 클래스에서 멤버를 선언하기 전에 명시해둔다
+ 아무것도 없으면 디폴트로 `private:`과 같이 즉, 전용 멤버로 선언된다

> getter와 setter

+ `private:`선언의 입출력
+ 접근자와 설정자

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

class car {
private:
	int speed;
public:
	int getspeed() {
		return speed;
	}
	void setspeed(int s) {
		speed = s;
	}
};

int main() {
	car mycar;
	mycar.setspeed(10);
	cout << mycar.getspeed() << endl;

	system("pause");
}

~~~
> 실행결과

~~~C++
10
계속하려면 아무 키나 누르십시오 . . .
~~~
> 멤버 함수의 외부 정의

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

class car {
private:
	int speed;
public:
	int getspeed();
	void setspeed(int s);
};

int main() {
	car mycar;
	mycar.setspeed(10);
	cout << mycar.getspeed() << endl;

	system("pause");
}

int car::getspeed() {
	return speed;
}
void car::setspeed(int s) {
	speed = s;
}
~~~
***
# 구조체

+ 서로 다른 타입의 변수들을 한곳에 모아 놓은 것
+ 클래스에서 멤버 함수를 제외하면 구조체다
+ 반대로 구조체에 함수를 추가하여 확장한 것이 클래스다

~~~C++
struct bank{
    int account;
    int balance;
    double rate;
};

bank a1;

a1.account = 10000;
~~~
