---
title: C++ (1)
date: 2020-07-06 10:10:44
categories:
- Etc.
tags:
- B.S. Course Work
- C, C++
---
# Day 1

## C와의 차이점

+ `#include <stdio.h>` -> `#include <iostream>`
  + 형식지정자 필요없어짐
+ `.h` 사라짐
  + 호환은 가능
+ `class`의 개념
  + object
+ `bool`의 개념
  + C
    + 참 : 0 이외의 값
  + C++
    + true : 1
    + false : 0
+ `string`의 개념

## Practice

> practice.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int a = 5;
    int b = 10;
    cout << "1. a + b / 3 * 3 = " << a + b / 3 * 3 << endl;
    cout << "2. b << 2 = " << (b << 2) << endl;
    cout << "3. a != b = " << (a != b) << endl;
    cout << "4. b % a = " << (b % a) << endl;
    cout << "5. (a > b) ? a : b = " << ((a > b) ? a : b) << endl;
    cout << "6. sizeof(a) = " << sizeof(a) << endl;
    int c;
    c = a++;
    cout << "7. C = a++ 이후 c의 값 : " << c << endl;
    a += b;
    cout << "8. a += b 이후 a의 값 : " << a << endl;
    cout << "9. a & b = " << (a & b) << endl;
    c = (a + b, a - b);
    cout << "10. c = (a + b, a - b) 이후 c의 값 : " << c << endl;
    return 0;
}
~~~

>> Output

~~~C++
1. a + b / 3 * 3 = 14
2. b << 2 = 40
3. a != b = 1
4. b % a = 0
5. (a > b) ? a : b = 10
6. sizeof(a) = 4
7. C = a++ 이후 c의 값 : 5
8. a += b 이후 a의 값 : 16
9. a & b = 0
10. c = (a + b, a - b) 이후 c의 값 : 6

Process finished with exit code 0
~~~

<!-- More -->

## 연산자

+ 산술 연산자
  + `*`, `/`, `%`
  + `+`, `-`
+ 비트 이동 연산자
  + `<<`, `>>`
+ 관계 연산자
  + `<`, `<=`, `>`, `>=`
  + `==`, `!=`
+ 논리 연산자
  + `&&`
  + `||`

## Plus

> plus.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int x = 10, y = 10;

    cout << "x = " << x << endl;
    cout << "++x = " << ++x << endl;
    cout << "x = " << x << endl;

    cout << "y = " << y << endl;
    cout << "y++ = " << y++ << endl;
    cout << "y = " << y << endl;

    return 0;
}
~~~

>> Output

~~~C++
x = 10
++x = 11
x = 11
y = 10
y++ = 10
y = 11

Process finished with exit code 0
~~~

## Switch

> switch.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int score;

    cout << "점수를 입력하세요 : ";
    cin >> score;

    score = score / 10;

    switch(score){
        case 10:
        case 9:
            cout << "A+ 입니다.";
            break;
        case 8:
            cout << "A 입니다.";
            break;
        case 7:
            cout << "B 입니다.";
            break;
        case 6:
            cout << "C 입니다.";
            break;
        default:
            cout << "F 입니다.";
            break;
    }

    return 0;
}
~~~

## Continue & Break

> continue_break.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int a;
    for(;;){
        cout << "정수 입력 : ";
        cin >> a;
        if(a == 0)
            break;
        if(a % 3 != 0){
            cout << "No" << endl;
            continue;
        }
        cout << "Yes" << endl;
    }

    return 0;
}
~~~

***

# Day 2

## Pointer

> pointer1.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int a = 10;
    int b = 12;
    int *pa; //Pointer 변수
    int *pb;
    pa = &a;
    pb = &b;
    cout << "Pointer of a : " << pa << " == " << &a << endl;
    cout << "Pointer of b : " << pb << " == " << &b << endl;

    return 0;
}
~~~

>> Output

~~~C++
Pointer of a : 0x7ffee6a4f958 == 0x7ffee6a4f958
Pointer of b : 0x7ffee6a4f954 == 0x7ffee6a4f954

Process finished with exit code 0
~~~

> pointer2.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int a[3] = {1, 2, 3};
    int *pa; //Pointer 변수
    pa = &a[0];
    cout << "*(pa + 1) = " << *(pa + 1) << endl;
    cout << "a[1] = " << a[1] << endl;

    return 0;
}
~~~

>> Output

~~~C++
*(pa + 1) = 2
a[1] = 2

Process finished with exit code 0
~~~

+ 포인터는 상수 - 변경 불가

## Function

> function.cpp

~~~C++
#include <iostream>
using namespace std;

int adder(int a, int b);

int main(){
    cout << adder(13, 14) << endl;
    return 0;
}

int adder(int a, int b){
    int sum;
    sum = a + b;
    return sum;
}
~~~

>> Output

~~~C++
27

Process finished with exit code 0
~~~

+ Call by Value
  + 값 복사 - 수정 불가
  + 전역변수 - 누구든지 접근 가능
+ Call by Address
+ Call by Reference

> call_by_value.cpp

~~~C++
#include <iostream>
using namespace std;

void swap(int x, int y);

int main(){
    int a = 2;
    int b = 3;

    cout << "a : " << a << "\tb : " << b <<endl;
    swap(a, b);
    cout << "a : " << a << "\tb : " << b <<endl;

    return 0;
}

void swap(int x, int y){
    int temp = x;
    x = y;
    y = temp;
}
~~~

>> Output

~~~C++
a : 2	b : 3
a : 2	b : 3

Process finished with exit code 0
~~~

> call_by_address.cpp

~~~C++
#include <iostream>
using namespace std;

void swap(int *x, int *y);

int main(){
    int a = 2;
    int b = 3;

    cout << "a : " << a << "\tb : " << b <<endl;
    swap(&a, &b);
    cout << "a : " << a << "\tb : " << b <<endl;

    return 0;
}

void swap(int *x, int *y){
    int temp = *x;
    *x = *y;
    *y = temp;
}
~~~

> call_by_reference.cpp

~~~C++
#include <iostream>
using namespace std;

void swap(int *x, int *y);

int main(){
    int a = 2;
    int b = 3;

    cout << "a : " << a << "\tb : " << b <<endl;
    swap(a, b);
    cout << "a : " << a << "\tb : " << b <<endl;

    return 0;
}

void swap(int &x, int &y){
    int temp = x;
    x = y;
    y = temp;
}
~~~

>> Output

~~~C++
a : 2	b : 3
a : 3	b : 2

Process finished with exit code 0
~~~

+ `swap(int, int)`는 기본 제공 함수

## 배열

~~~C++
int n[10]; // 정수 10개짜리 빈 메모리 공간
double d[] = [0.1, 0.2, 0.5, 3.9]; // d의 크기는 자동 4로 설정
n[10] = 20; // 인덱스 0~9까지만
n[-1] = 9.9; // 인덱스 음수 불가
int m[2][5]; // 2행 5열의 2차원 배열 선언
int grade[2][3] = {{10, 20, 30}, {40, 50, 60}};
~~~

+ 배열은 0부터 시작
+ 함수에 매개변수로 전달 시 배열의 크기도 함께 전달

> two_dimension_array.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int Ary[5][5];
    int i, j;
    for(i = 0; i < 5; i++){
            for(j = 0; j < 5; j++){
                if(i >= j){
                    Ary[i][j] = i + 1;
                }
                else{
                    Ary[i][j] = 0;
                }
            }
    }
    for(i = 0; i < 5; i++){
        for(j = 0; j < 5; j++){
            cout << Ary[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
~~~

>> Output

~~~C++
1 0 0 0 0 
2 2 0 0 0 
3 3 3 0 0 
4 4 4 4 0 
5 5 5 5 5 

Process finished with exit code 0
~~~

## Calling a Function with an Array as a Parameter

~~~C++
#include <iostream>
using namespace std;

int addArray(int *a, int size);
void makeDouble(int a[], int size);
void printArray(int a[], int size);

int main(){
    int n[] = {1, 2, 3, 4, 5};

    int sum = addArray(n, 5);
    cout << "배열 n의 합은 " << sum << "입니다." << endl;
    makeDouble(n, 5);
    printArray(n, 5);

    return 0;
}

int addArray(int *a, int size){
    int su = 0;
    for(int i = 0; i < size; i++){
        su = su + a[i];
    }
    return su;
}
void makeDouble(int a[], int size){
    for(int i = 0; i < size; i++){
        a[i] = a[i] * a[i];
    }
}
void printArray(int a[], int size){
    for(int i = 0; i < size; i++){
        cout << a[i] << "\t";
    }
}
~~~

>> Output

~~~C++
배열 n의 합은 15입니다.
1	4	9	16	25	
Process finished with exit code 0
~~~

+ 배열이 함수의 파라미터로 들어갈 경우
  + `int *a`
  + `int a[]`

## Const and Pointer

> const_pointer.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int var1 = 1;
    int var2 = 2;

    const int *p1 = &var1;
    int *const p2 = &var2;

    //*p1 = 10; 오류
    var1 = 10;
    *p2 = 3;

    cout << var1 << endl;
    cout << var2 << endl;

    return 0;
}
~~~

>> Output

~~~C++
10
3

Process finished with exit code 0
~~~

+ `const int *pNum` - 포인터 자체의 값이 상수
+ `int *const pNum` - 포인터가 가리키는 값이 상수

> example.cpp

~~~C++
#include <iostream>
using namespace std;

#define N 4

void print_arr(int *arr);
void percentage(int *arr);

int main(){
    int count[N] = {42, 37, 83, 33};
    cout << "인원수 : ";
    print_arr(count);
    percentage(count);
    cout << "\n백분율 : ";
    print_arr(count);

    return 0;
}

void print_arr(int *arr){
    for(int i = 0; i < N; i++){
        cout << *(arr + i) << "\t";
    }
}
void percentage(int *arr){
    int sum = 0;
    for(int i = 0; i < N; i++){
        sum = sum + *(arr + i);
    }
    for(int i = 0; i < N; i++){
        *(arr + i) = *(arr + i) * 100 / sum;
    }
}
~~~

>> Output

~~~C++
인원수 : 42	37	83	33	
백분율 : 21	18	42	16	
Process finished with exit code 0
~~~

## String

> getline.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    string song("Falling in love with you");
    string elvis("Elvis Presley");
    string singer;

    cout << song + "를 부른 가수는";
    cout << "(힌트 : 첫글자는 " << elvis[0] << ")?";

    getline(cin, singer);
    if(singer == elvis)
        cout << "맞았습니다." << endl;
    else
        cout << "틀렸습니다. " + elvis + "입니다." << endl;

    return 0;
}
~~~

>> Output

~~~C++
Falling in love with you를 부른 가수는(힌트 : 첫글자는 E)?Elvis Presley
맞았습니다.

Process finished with exit code 0

...

Falling in love with you를 부른 가수는(힌트 : 첫글자는 E)?asdf
틀렸습니다. Elvis Presley입니다.

Process finished with exit code 0
~~~

+ `char` - `' '`
+ `string` - `" "`

## Quiz

> quiz1.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    string str;
    cout << "문자열 입력>>";

    getline(cin, str);
    int len = str.size();
    for(int i = 1; i <= len; i++){
        for(int j = 1; j <= i; j++){
            cout << str[j-1];
        }
        cout << endl;
    }

    return 0;
}
~~~

>> Output

~~~C++
문자열 입력>>Morning
M
Mo
Mor
Morn
Morni
Mornin
Morning

Process finished with exit code 0
~~~

> quiz2.cpp

~~~C++
#include <iostream>
using namespace std;

double biggest(double x[], int n){
    double big = x[0];
    for(int i = 0; i < n; i++){
        if(x[i] > big)
            big = x[i];
    }
    return big;
}

int main(){
    double a[5];
    cout << "5개의 실수를 입력하라>>";
    for(int i = 0; i < 5; i++)
        cin >> a[i];
    cout << "제일 큰 수 = " << biggest(a, 5) << endl;

    return 0;
}
~~~

>> Output

~~~C++
5개의 실수를 입력하라>>20.0
3.44
44.66
22.0
40.0
제일 큰 수 = 44.66

Process finished with exit code 0
~~~

***

# Day 3

## Namespace

~~~C++
namespace name{ // name이라는 이름 공간 생성
    // 이곳에 선언된 모든 이름은 name 이름 공간에 생성된 이름
    void function(){
        ...
    }
    ...
}
name::function()
~~~

+ 이름(Identifier) 충돌이 발생하는 경우
  + 여러 명이 서로 나누어 프로젝트를 개발하는 경우
  + 오픈 소스 혹은 다른 사람이 작성한 소스나 목적 파일을 가져와서 컴파일하거나 링크하는 경우
+ `namespace`
  + 이름 충돌 해결
  + 개발자가 자신만의 이름 공간을 생성할 수 있도록 함
+ `std`
  + ANSI C++ 표준에서 정의한 이름 공간(`namespace`) 중 하나
  + `<iostream>`

> namespace.cpp

~~~C++
#include <iostream>
using namespace std;

namespace Graphic{
    int maximum = 100;
}
namespace Math{
    int maximum = 65536;
    int add(int a, int b){return a + b;}
    int sub(int a, int b){return a - b;}
}

int main(){
    cout << "Radius Maximum = " << Graphic::maximum << endl;
    cout << "Integer Maximum = " << Math::maximum << endl;
    cout << "Integer Add = " << Math::add(2,4) << endl;
    cout << "Integer Sub = " << Math::sub(2,4) << endl;

    return 0;
}
~~~

>> Output

~~~C++
Radius Maximum = 100
Integer Maximum = 65536
Integer Add = 6
Integer Sub = -2

Process finished with exit code 0
~~~

## String

+ `cin.getline(char buf[], int size, char delimitChar)`
  + 공백이 낀 문자열을 입력 받는 방법
  + `buf`에 최대 `size-1`개의 문자 입력(끝에 `\0` 붙임)
  + `delimitChar`를 만나면 입력 중단(끝에 `\0` 붙임)
    + Default : `\n`(Enter)
+ `string` 클래스
  + 문자열의 크기에 따른 제약 없음
    + 스스로 문자열의 버퍼 조정
  + 문자열 복사, 비교, 수정 등을 위한 다양한 함수와 연산자 제공
  + 객체 지향적
  + `<string>` 헤더 파일에 선선

> cin_getline.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    cout << "주소 입력 : ";

    char address[100];
    cin.getline(address, 100, '\n');
    cout << "주소는 " << address << "입니다.\n";

    return 0;
}
~~~

>> Output

~~~C++
주소 입력 : ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ
주소는 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ입니다.

Process finished with exit code 0
~~~

> string_getline.cpp

~~~C++
#include <iostream>
#include <string>
using namespace std;

int main(){
    cout << "주소 입력 : ";

    string address;
    getline(cin, address);
    cout << "주소는 " << address << "입니다.\n";

    return 0;
}
~~~

>> Output

~~~C++
주소 입력 : asdfasdfasdfasdfasdfasdfasdfa
주소는 asdfasdfasdfasdfasdfasdfasdfa입니다.

Process finished with exit code 0
~~~

> search.cpp

~~~C++
#include <iostream>
#include <string>
using namespace std;

int main(){
    string str;
    int count = 0;
    cout << "문자들을 입력하라 : ";
    getline(cin, str);

    int i = 0;
    while(true){
        if(str[i] == 'o'){
            count += 1;
        }
        else if(str[i] == '\0'){
            break;
        }
        i += 1;
    }

    cout << 'o' << "의 개수는 " << count << endl;

    return 0;
}
~~~

>> Output

~~~C++
문자들을 입력하라 : Hello, World!
o의 개수는 2

Process finished with exit code 0
~~~

## Class & Object

~~~C++
class Name{
public:
    int value;
    void function();
    ...
};

void Name::function(){
    ...
}
~~~

+ `class`
  + 상태 정보(속성) : 멤버 변수
  + Action : 멤버 함수
  + 객체를 만들어내기 위해 정의된 설계도, 틀
  + 클래스는 객체가 아니며 실체도 아님
+ Object
  + 객체는 생성될 때 클래스의 모양을 그대로 가지고 탄생
  + 멤버 변수와 멤버 함수로 구성
  + 메모리에 생성, 실체(Instance)로도 불림
+ Why `class`?
  + Encapsulation
    + `public`
      + 외부에서 접근 가능
    + `protectected`
    + `private`
      + 외부에서 접근 불가
      + Default
    + 캡슐화
    + 클래스로 묶어둠
  + Inheritance
  + Polymorphism

> circle.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
public:
    int radius;
    double getArea();
};
double Circle::getArea(){
    return 3.14*radius*radius;
}

int main(){
    Circle donut;
    donut.radius = 1;
    double area = donut.getArea();
    cout << "donut 면적은 " << area << "입니다." << endl;

    Circle pizza;
    pizza.radius = 30;
    area = pizza.getArea();
    cout << "pizza 면적은 " << area << "입니다." << endl;

    return 0;
}
~~~

>> Output

~~~C++
donut 면적은 3.14입니다.
pizza 면적은 2826입니다.

Process finished with exit code 0
~~~

> rectangle.cpp

~~~C++
#include <iostream>
using namespace std;

class Rectangle{
public:
    int width;
    int height;
    int getArea(){
        return width*height;
    }
};

int main(){
    Rectangle rect;
    rect.width = 3;
    rect.height = 5;
    cout << "사각형의 면적은 " << rect.getArea() << endl;

    return 0;
}
~~~

>> Output

~~~C++
사각형의 면적은 15

Process finished with exit code 0
~~~

## Constructor

+ 생성자
  + 객체가 생성되는 시점에서 자동으로 호출되는 멤버 함수
  + 생성자는 클래스의 이름과 같음
  + `public`으로 선언
+ 종류
  + 기본 생성자
    + 매개변수 없음
    + 생성자가 없으면 컴파일러가 자동으로 만들어줌
  + 사용자 정의 생성자
    + 매개변수 존재
    + 객체 생성 및 멤버 변수 초기값 정의
    + 기본 생성자 자동 생성 X

> constructor.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
private:
    int radius;
public:
    Circle();
    Circle(int r);
    double getArea();
};

int main(){
    Circle donut;
    double area = donut.getArea();
    cout << "donut 면적은 " << area << "입니다." << endl;

    Circle pizza(30);
    area = pizza.getArea();
    cout << "pizza 면적은 " << area << "입니다." << endl;

    return 0;
}

double Circle::getArea(){
    return 3.14*radius*radius;
}

Circle::Circle(){
    radius = 1;
}

Circle::Circle(int r){
    radius = r;
}
~~~

>> Output

~~~C++
donut 면적은 3.14입니다.
pizza 면적은 2826입니다.

Process finished with exit code 0
~~~

## Quiz

> quiz2.cpp

~~~C++
#include <iostream>
using namespace std;

class Tower{
private:
    int height;
public:
    Tower();
    Tower(int h);
    int getHeight();
};

int main(){
    Tower myTower;
    Tower seoulTower(100);
    cout << "높이는 " << myTower.getHeight() << "미터" << endl;
    cout << "높이는 " << seoulTower.getHeight() << "미터" <<endl;
    return 0;
}

Tower::Tower(){
    height = 1;
}
Tower::Tower(int h){
    height = h;
}
int Tower::getHeight(){
    return height;
}
~~~

>> Output

~~~C++
높이는 1미터
높이는 100미터

Process finished with exit code 0
~~~

***

# Day 4

## Class Practice

> integer.cpp

~~~C++
#include <iostream>
using namespace std;

class Integer{
private:
    int val1;
public:
    Integer(int val);
    Integer(string val);
    void set(int val);
    int get();
    bool isEven();
};

int main(){
    Integer n(30);
    cout << n.get() << ' ';
    n.set(50);
    cout << n.get() << ' ';
    Integer m("300");
    cout << m.get() << ' ';
    cout << m.isEven();

    return 0;
}

Integer::Integer(int val){
    val1 = val;
}
Integer::Integer(string val){
    val1 = stoi(val);
}
void Integer::set(int val){
    val1 = val;
}
int Integer::get(){
    return val1;
}
bool Integer::isEven(){
    if(val1%2 == 0){
        return true;
    }
    else{
        return false;
    }
}
~~~

>> Output

~~~C++
30 50 300 1
Process finished with exit code 0
~~~

## Destructor

~~~C++
class name{
    ~name();
}
name::~name(){};
~~~

+ 객체가 소멸되는 시점에서 자동으로 호출되는 함수
+ 소멸자의 목적
  + 객체가 사라질 때 마무리 작업을 위함
  + 실행 도중 동적으로 할당 받은 메모리 해제, 파일 저장 및 닫기, 네트워크 닫기 등
+ 소멸자는 `return` 불가
+ 객체는 생성의 반대순으로 소멸
+ 동적 메모리 할당 사용 시 소멸자 이용

> destructor.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
public:
    Circle();
    Circle(int r);
    int radius;
    double getArea();
    ~Circle();
};

int main(){
    Circle donut;
    double area = donut.getArea();
    cout << "donut 면적은 " << area << "입니다." << endl;
    donut.~Circle();

    Circle pizza(30);
    area = pizza.getArea();
    cout << "pizza 면적은 " << area << "입니다." << endl;
    pizza.~Circle();

    return 0;
}

Circle::Circle(){
    radius = 1;
    cout << "반지름 " << radius << " 원 생성" << endl;
}
Circle::Circle(int r){
    radius = r;
    cout << "반지름 " << radius << " 원 생성" << endl;
}
double Circle::getArea(){
    return 3.14*radius*radius;
}
Circle::~Circle(){
    cout << "반지름 " << radius << " 원 소멸" << endl;
}
~~~

>> Output

~~~C++
반지름 1 원 생성
donut 면적은 3.14입니다.
반지름 1 원 소멸
반지름 30 원 생성
pizza 면적은 2826입니다.
반지름 30 원 소멸
반지름 30 원 소멸
반지름 1 원 소멸

Process finished with exit code 0
~~~

## 접근 지정자

+ 캡슐화의 목적
  + 객체 보호, 보안
  + C++에서 객체의 캡슐화 전략
    + 객체의 상태를 나타내는 데이터 멤버(멤버 변수)에 대한 보호
    + 중요한 멤버는 다른 클래스나 객체에서 접근할 수 없도록 보호
    + 외부와으이 인터페이스를 위해서 일부 멤버는 외부에 접근 허용
+ 멤버에 대한 3가지 접근 지정자
  + `private`
    + 동일한 클래스의 멤버 함수에만 제한함
  + `public`
    + 모든 다른 클래스에 허용, 외부함수(`main()`)도 허용
  + `protected`
    + 클래스 자신과 상속받은 자식 클래스에만 허용

## Inline

+ 짧은 함수의 다수 호출로 인해 오버헤드가 일어남
+ 이를 `inline`을 통해 해결
+ 장점 : 시간 단축
+ 단점 : 코드가 길어짐
+ 클래스의 선언부에 구현된 멤버 함수는 자동 `inline` 함수

## Program

+ 바람직한 프로그램
  + `main.cpp`
  + `class.h` : 클래스 선언부 - 헤더 파일(`.h`)에 저장
  + `class.cpp` : 클래스 구현부 - `.cpp` 파일에 저장
+ 클래스를 헤더 파일과 `.cpp` 파일로 분리하여 작성
+ 목적 : 클래스 재사용

> main.cpp

~~~C++
#include <iostream>
#include "Box.h"
using namespace std;

int main(){
    Box b(10, 2);
    b.draw();
    cout << endl;
    b.setSize(7, 4);
    b.setFill('%');
    b.draw();

    return 0;
}
~~~

> Box.h

~~~C++
#ifndef UNTITLED_BOX_H // 조건 컴파일 역할, n은 not, 정의되어있지 않으면 정의, 정의되어있으면 실행 X
#define UNTITLED_BOX_H // 중복 include 실행방지

class Box {
private:
    int width, height;
    char fill;
public:
    Box(int w, int h);
    void setFill(char f);
    void setSize(int w, int h);
    void draw();
};

#endif // UNTITLED_BOX_H
~~~

> Box.cpp

~~~C++
#include <iostream>
#include "Box.h"
using namespace std;

Box::Box(int w, int h){
    setSize(w, h);
    fill = '*';
}

void Box::setFill(char f){
    fill = f;
}

void Box::setSize(int w, int h){
    width = w;
    height = h;
}

void Box::draw(){
    for(int n = 0; n < height; n++){
        for(int m = 0; m < width; m++)
            cout << fill;
        cout << endl;
    }
}
~~~

>> Output

~~~C++
**********
**********

%%%%%%%
%%%%%%%
%%%%%%%
%%%%%%%

Process finished with exit code 0
~~~

## Random

~~~C++
#include <cstdlib>
#include <ctime>

int rand(void);
void srand((unsigned int) time(NULL));

srand((unsigned) time(0)); // Seed 값 설정, 현재시간 이용 초기화
rand() % (End - Begin + 1) + Begin; // int 스케일링
rand() % (End - Begin) / RAND_MAX + Begin; // float 스케일링
~~~

+ `rand()` = $[0, 2^{15}-1]$

> random.cpp

~~~C++
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

class EvenRandom{
public:
    EvenRandom();
    int next();
    int nextinRange(int low, int high);
};

int main(){
    EvenRandom r;
    cout << "-- 0에서 " << RAND_MAX << "까지의 랜덤 짝수 정수 10개 : " << endl;
    for(int i = 0; i < 10; i++){
        int n = r.next();
        cout << n << '\t';
    }
    cout << endl << endl << "-- 2에서 10까지의 랜덤 짝수 정수 10개 : " << endl;
    for(int i = 0; i < 10; i++){
        int n = r.nextinRange(2, 10);
        cout << n << '\t';
    }

    return 0;
}

EvenRandom::EvenRandom(){
    srand((unsigned int) time(0));
}

int EvenRandom::next(){
    int ran = rand();
    while(true){
        if(ran % 2 == 0){
            return ran;
        }
        ran = rand();
    }
}

int EvenRandom::nextinRange(int low, int high){
    int ran = rand() % (high - low + 1) + low;
    while(true){
        if(ran % 2 == 0){
            return ran;
        }
        ran = rand() % (high - low + 1) + low;
    }
}
~~~

>> Output

~~~C++
-- 0에서 2147483647까지의 랜덤 짝수 정수 10개 : 
754544228	1733691306	1091657446	454057686	268781690	2138855454	962661234	305563340	433726818	1087132208	

-- 2에서 10까지의 랜덤 짝수 정수 10개 : 
10	10	6	2	2	4	6	4	2	4	
Process finished with exit code 0
~~~