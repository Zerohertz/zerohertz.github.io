---
title: C++ (2)
date: 2019-01-06 15:34:50
categories:
- Etc.
tags:
- C, C++
---
# 배열

> 배열의 선언

~~~C++
int grad[10]; // 자료형, 배열 이름, 배열 크기
~~~
> 배열의 초기화

~~~C++
int grade[5] = {10, 20, 30, 40, 50};
~~~
+ 만약 배열의 크기가 초기값의 개수보다 크다면 나머지는 `0`으로 초기화 된다.
+ 배열의 크기를 선언하지 않으면 초기값의 개수에 맞춰서 선언된다.

> 배열의 복사

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

int main() {
	const int SIZE = 5;
	int i;
	int a[SIZE] = { 1, 2, 3, 4, 5 };
	int b[SIZE];

	for (i = 0; i < SIZE; i++)
		b[i] = a[i];
	for (i = 0; i < SIZE; i++)
		cout << "b[" << i << "] = " << b[i] << endl;
	system("pause");
}
~~~

<!-- more -->

> 실행결과

~~~C++
b[0] = 1
b[1] = 2
b[2] = 3
b[3] = 4
b[4] = 5
계속하려면 아무 키나 누르십시오 . . .
~~~
> 다차원 배열

~~~C++
int s[10]; // 1차원 배열
int s[3][10]; // 2차원 배열
int s[5][3][10]; // 3차원 배열

int s[3][5] = {
	{0,1,2,3,4},
	{10,11,12,13,14},
	{20,21,22,23,24}
};
~~~
+ 혹은 n차원 배열을 n중 루프를 통하여 데이터를 처리할 수 있다.
***
# 포인터

> 메모리의 주소를 가지고 있는 변수.

+ `&`를 붙임으로써 변수의 주소를 알 수 있다.

~~~C++
cout << "i의 주소: " << &i << endl;
~~~
> 포인터의 선언

~~~C++
int *p; // 정수를 가리키는 포인터
char *pc; // 문자를 가리키는 포인터
float *pf; // 실수(float 형)를 가리키는 포인터
double *pd; // 실수(double 형)를 가리키는 포인터
~~~
> 포인터와 변수의 연결

~~~C++
int *p = &i; // 변수 i의 주소가 포인터 p로 대입
~~~
+ 포인터 초기화시 `NULL`로 한다.

> 포인터의 용도

+ 동적으로 할당된 메모리를 사용하는 경우
+ 함수의 매개 변수로 변수의 주소를 전달하는 경우
+ 클래스의 멤버 변수나 멤버 함수를 호출하는 경우

> 동적 메모리 할당

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

int main() {
	int *pi;

	pi = new int;
	cout << "pi = " << pi << endl << "*pi = " << *pi << endl;
	
	*pi = 100;
	cout << "pi = " << pi << endl << "*pi = " << *pi << endl;

	delete pi;
	cout << "pi = " << pi << endl << "*pi = " << *pi << endl;

	system("pause");
}
~~~
![delete한 뒤의 상황](https://user-images.githubusercontent.com/42334717/50733545-d2443300-11d2-11e9-981a-0a81acd87eb5.png)
~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

int main() {
	int *pi;

	pi = new int;
	cout << "pi = " << pi << endl << "*pi = " << *pi << endl;
	
	*pi = 100;
	cout << "pi = " << pi << endl << "*pi = " << *pi << endl;

	delete pi;

	system("pause");
}
~~~
> 실행결과

~~~C++
pi = 00824278
*pi = -842150451
pi = 00824278
*pi = 100
계속하려면 아무 키나 누르십시오 . . .
~~~
+ `new`로 동적 메모리 할당
+ `*pi = 100;`에서 동적 메모리 사용
+ `delete`로 동적 메모리 반납

> const 포인터

~~~C++
const int *p1;
int *const p2;
~~~
+ 위의 상황은 p1이 가리키는 내용이 정수형 상수이다
+ 아래의 상황은 p2가 상수라는 의미이다

> swap을 통한 참조에 의한 호출 예제

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

void swap(int *px, int *py);

int main() {
	int a = 100, b = 200;
	cout << "swap() 호출전: a = " << a << ", b = " << b << endl;
	swap(&a, &b);
	cout << "swap() 호출후: a = " << a << ", b = " << b << endl;

	system("pause");
}

void swap(int *px, int *py) {
	int tmp;
	cout << "In swap() : *px = " << *px << ", *py = " << *py << endl;
	tmp = *px; *px = *py; *py = tmp;
	cout << "In swap() : *px = " << *px << ", *py = " << *py << endl;
}
~~~
> 실행결과

~~~C++
swap() 호출전: a = 100, b = 200
In swap() : *px = 100, *py = 200
In swap() : *px = 200, *py = 100
swap() 호출후: a = 200, b = 100
계속하려면 아무 키나 누르십시오 . . .
~~~
