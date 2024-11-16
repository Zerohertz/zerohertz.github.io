---
title: C++ (1)
date: 2019-01-06 00:29:57
categories:
- Etc.
tags:
- C, C++
---
# 자료형

+ `short` - short형 정수
+ `int` - 정수
+ `long` -  long형 정수
+ `unsigned short` - 부호없는 short형 정수
+ `unsigned int` - 부호없는 정수
+ `unsigned long` - 부호없는 long형 정수
+ `char` - 문자 및 정수
+ `unsigned char` - 문자 및 부호없는 정수
+ `float` - 단일정밀도 부동소수점
+ `double` - 두배정밀도 부동소수점
+ `bool` - True or False
<!-- more -->
***
# 입출력

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

int main()
{
	int example = 10;
	cout << "example = " << example << endl;
	cout << example;
	cout << example << endl << endl;

	system("pause");
	return 0;
}
~~~
> 실행결과

~~~C++
example = 10
1010

계속하려면 아무 키나 누르십시오 . . .
~~~
+ `endl`은 줄바꿈이다!(C에서의 `\n`)
+ `cin`, `cout`들은 각각 `input`, `output`이다.

~~~C++
int grade[5];
cout << "ㅁㄴㅇㄹ" << endl;
cin >> greade[i];
~~~
***
# if문

~~~C++
if(조건식){
    문장;
}
else if(조건식){
    문장;
}
else(조건식){
    문장;
}
~~~
***
# while문

~~~C++
while(조건식){
    반복문장;
}
~~~
***
# do-while문

~~~C++
do{
    반복문장;
}while(조건식);
~~~
***
# for문

~~~C++
for(초기식; 조건식; 증감식){
    문장;
}
~~~
***
# 함수

~~~C++
#include <iostream>
#include <windows.h>
using namespace std;

int square(int n) {
	return n * n;
}

int main() {
	cout << square(99) << endl;
	system("pause");
}
~~~
> 실행결과

~~~C++
9801
계속하려면 아무 키나 누르십시오 . . .
~~~
> 디폴트 매개 변수

~~~C++
void sub(int p1, int p2, int p3 = 30);
~~~
+ 위와 같이 함수를 정의하면 매개 변수를 전달하지 않아도 디폴트 값을 대신 넣어준다
+ 디폴트 매개 변수는 뒤에서 앞으로 정의해야한다

> 중복 함수

+ 매개 변수의 개수와 타입으로 구분
+ 반환형이 다르다고 중복할 수는 없음

***
# 저장 유형 지정자(storage class specifier)

+ `auto`(자동 변수)
+ `register`(레지스터 변수)
+ `static`(정적 변수)
+ `extern`(외부의 정적 변수)
