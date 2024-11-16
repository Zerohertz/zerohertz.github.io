---
title: C++ (7)
date: 2020-07-06 10:10:46
categories:
- Etc.
tags:
- B.S. Course Work
- C, C++
---
# Day 9

## Polymorphism(다형성)

+ Overloading
  + 함수 중복
  + 연산자 중복
  + Default Parameter
+ Overriding
  + 함수 재정의

## Function Overloading

+ 다른 함수로 인식
+ 함수의 이름 동일
+ 함수의 매개변수 type, 개수 다름
+ return type 무관
+ 소멸자 불가 - 매개변수 X
+ 모호하지 않게 선언

<!-- More -->

> overloading.cpp

~~~C++
#include <iostream>
using namespace std;

int big(int a, int b){
    if(a > b) return a;
    else return b;
}

int big(int a[], int size){
    int res = a[0];
    for(int i = 1; i < size; i++)
        if(res < a[i]) res = a[i];
    return res;
}

int main(){
    int array[5] = {1, 9, -2, 8, 6};
    cout << big(2, 3) << endl;
    cout << big(array, 5) << endl;
    
    return 0;
}
~~~

>> Output

~~~C++
3
9

Process finished with exit code 0
~~~

## Default Parameter

+ 사전에 값을 선언한 함수의 매개변수
+ 생략 가능
+ 일반 매개변수 뒤에 존재

> default_param.cpp

~~~C++
#include <iostream>
using namespace std;

void f(char c=' ', int line = 1);

int main(){
    f();
    f('%');
    f('@', 5);

    return 0;
}

void f(char c, int line){
    for(int i = 0; i < line; i++){
        for(int j = 0; j < 10; j++){
            cout << c;
        }
        cout << endl;
    }
}
~~~

>> Output

~~~C++
          
%%%%%%%%%%
@@@@@@@@@@
@@@@@@@@@@
@@@@@@@@@@
@@@@@@@@@@
@@@@@@@@@@

Process finished with exit code 0
~~~

> myvec.cpp

~~~C++
#include <iostream>
using namespace std;

class MyVector{
    int *p;
    int size;
public:
    MyVector(int n = 100){p = new int[n]; size = n;}
    ~MyVector(){delete [] p;}
};

int main(){
    MyVector *v1, *v2;
    v1 = new MyVector();
    v2 = new MyVector();

    delete v1;
    delete v2;

    return 0;
}
~~~

>> Output

~~~C++
Process finished with exit code 0
~~~

## Static & Non-static

+ `static`
  + 변수와 함수에 대한 기억 부류의 한 종류
    + 생명 주기 : 프로그램이 시작될 때 생성, 프로그램 종료 시 소멸
    + 사용 범위 : 선언된 범위, 접근 지정에 따름
  + 전역 변수나 전역변수를 클래스에 캡슐화
    + 전역 변수나 전역 함수를 가능한 사용하지 않도록
    + 전역 변수나 전역 함수를 `static`으로 선언하여 클래스 멤버로 선언
  + 객체 사이에 공유 변수를 만들고자 할 때
    + `static` 멤버를 선언하여 모든 객체들이 공유
+ 클래스의 멤버
  + `static`
    + 프로그램이 시작할 때 생성
    + 클래스당 한번만 생성, 클래스 멤버라고 불림
    + 클래스의 모든 인스턴스(객체)들이 공유하는 멤버
  + non-`static`
    + 객체가 생성될 때 함께 생성
    + 객체마다 객체 내에 생성
    + 인스턴스 멤버라고 불림

> person.cpp

~~~C++
#include <iostream>
using namespace std;

class Person{
public:
    double money;
    void addMoney(int money){
        this->money += money;
    }
    static int sharedMoney;
    static void addShared(int n){
        sharedMoney += n;
    }
};

int Person::sharedMoney = 10;

int main(){
    Person han;
    han.money = 100;
    han.sharedMoney = 200;

    Person lee;
    lee.money = 150;
    lee.addMoney(200);
    lee.addShared(200);

    cout << han.money << '\t' << lee.money << endl;
    cout << han.sharedMoney << '\t' << lee.sharedMoney << endl;

    Person::sharedMoney = 1000;
    Person::addShared(20000);

    cout << han.money << '\t' << lee.money << endl;
    cout << han.sharedMoney << '\t' << lee.sharedMoney << endl;

    return 0;
}
~~~

>> Output

~~~C++
100	350
400	400
100	350
21000	21000

Process finished with exit code 0
~~~

> employee.cpp

~~~C++
#include <iostream>
using namespace std;

class Employee{
    string name;
    double salary;
    int static count;
public:
    Employee(string name = "", double salary = 0):name(name), salary(salary){
        this->count++;
    }
    int static getCount(){
        return count;
    }
    ~Employee(){
        this->count--;
    }
};

int Employee::count = 0;

int main(){
    Employee e1("김철수");
    Employee e2;
    Employee e3("김철호", 20000);

    int n = Employee::getCount();
    cout << "현재의 직원 수 : " << n << endl;

    return 0;
}
~~~

>> Output

~~~C++
현재의 직원 수 : 3

Process finished with exit code 0
~~~

> circle.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
private:
    static int numOfCircles;
    int radius;
public:
    Circle(int r = 1):radius(r){numOfCircles++;}
    ~Circle(){numOfCircles--;}
    double getArea(){return 3.14*radius*radius;}
    static int getNumOfCircles(){return numOfCircles;}
};

int Circle::numOfCircles = 0;

int main(){
    Circle *p = new Circle[10];
    cout << "할당된 원의 개수 : " << Circle::getNumOfCircles() << endl;

    delete [] p;
    cout << "할당된 원의 개수 : " << Circle::getNumOfCircles() << endl;

    Circle a;
    cout << "할당된 원의 개수 : " << Circle::getNumOfCircles() << endl;

    Circle b;
    cout << "할당된 원의 개수 : " << Circle::getNumOfCircles() << endl;

    return 0;
}
~~~

>> Output

~~~C++
할당된 원의 개수 : 10
할당된 원의 개수 : 0
할당된 원의 개수 : 1
할당된 원의 개수 : 2

Process finished with exit code 0
~~~

### Timeline of Program

1. 프로그램 시작
   + 전역 변수
   + `static` 멤버
     + 멤버 변수
     + 멤버 함수
2. 객체
   + non-`static` 멤버
3. 객체 종료
   + non-`static` 멤버 종료
4. 프로그램 끝
   + 전역 변수 종료
   + `static` 멤버 종료

### Access

+ `static` 멤버 함수 -> `static` 멤버 변수 : 가능
+ `static` 멤버 함수 -> non-`static` 멤버 변수 : 불가능
+ `static` 멤버 함수 -> non-`static` 멤버 함수 : 불가능
+ non-`static` 멤버 함수 -> non-`static` 멤버 변수 : 가능
+ non-`static` 멤버 함수 -> `static` 멤버 변수 : 가능
+ non-`static` 멤버 함수 -> `static` 멤버 함수 : 가능
+ `static` 멤버 함수가 접근할 수 있는 것
  + `static` 멤버 함수
  + `static` 멤버 변수
  + 함수 내의 지역 변수

~~~C++
this->sharedMoney += n; // static 이후 객체 생성 - 오류
~~~

## Quiz

> quiz.cpp

~~~C++
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

class Random{
public:
    void static seed(){srand(time(NULL));}
    int static nextInt(int start, int end);
    char static nextAlphabet();
    double static nextDouble();
};

int main(){
    Random::seed();
    cout << "1에서 100까지 랜덤한 정수 10개를 출력합니다." << endl;
    for(int i = 0; i < 10; i++) cout << Random::nextInt(1, 100) << '\t';
    cout << endl;

    cout << "알파벳을 랜덤하게 10개를 출력합니다." << endl;
    for(int i = 0; i < 10; i++) cout << Random::nextAlphabet() << '\t';
    cout << endl;

    cout << "랜덤한 실수를 10개 출력합니다." << endl;
    for(int i = 0; i < 5; i++) cout << Random::nextDouble() << '\t';
    cout << endl;
    for(int i = 0; i < 5; i++) cout << Random::nextDouble() << '\t';

    return 0;
}

int Random::nextInt(int start, int end){
    return rand() % (end - start + 1) + start;
}
char Random::nextAlphabet(){
    int num;
    while(true){
        num = nextInt(65, 122);
        if(num >= 91 && num <= 96){
            continue;
        }
        else{
            return num;
        }
    }
}
double Random::nextDouble(){
    return (double) rand() / RAND_MAX;
}
~~~

>> Output

~~~C++
1에서 100까지 랜덤한 정수 10개를 출력합니다.
47	98	79	68	45	18	87	76	88	15	
알파벳을 랜덤하게 10개를 출력합니다.
l	K	W	k	d	d	y	f	X	X	
랜덤한 실수를 10개 출력합니다.
0.891485	0.195258	0.707381	0.95967	0.167292	
0.673519	0.836099	0.318814	0.307098	0.397495
~~~

***

# Day 10

## Friend

+ 클래스의 멤버 함수가 아닌 외부 함수
  + 전역 함수
  + 다른 클래스의 멤버 함수
+ `friend`로 클래스 내에 선언된 함수
  + 클래스의 모든 멤버를 접근할 수 있는 권한 부여
  + 프렌드 함수라고 부름
+ friend
  + 전역 함수
  + 다른 클래스의 멤버 함수
  + 다른 클래스 전체

> friend1.cpp

~~~C++
#include <iostream>
using namespace std;

class Rect;
bool equals(Rect r, Rect s);

class Rect{
    int width, height;
public:
    Rect(int width, int height):width(width), height(height){};
    friend bool equals(Rect r, Rect s);
};

int main(){
    Rect a(3, 4), b(4, 5);
    if(equals(a, b)) cout << "equal" << endl;
    else cout << "not equal" << endl;

    return 0;
}

bool equals(Rect r, Rect s){
    if(r.width == s.width && r.height == s.height) return true;
    else return false;
}
~~~

> friend2.cpp

~~~C++
#include <iostream>
using namespace std;

class Rect;
bool equals(Rect r, Rect s);

class RectManager{
public:
    bool equals(Rect r, Rect s);
};

class Rect{
    int width, height;
public:
    Rect(int width, int height):width(width), height(height){};
    friend bool RectManager::equals(Rect r, Rect s);
};

int main(){
    Rect a(3, 4), b(4, 5);
    RectManager Man;
    if(Man.equals(a, b)) cout << "equal" << endl;
    else cout << "not equal" << endl;

    return 0;
}

bool RectManager::equals(Rect r, Rect s){
    if(r.width == s.width && r.height == s.height) return true;
    else return false;
}
~~~

> friend3.cpp

~~~C++
#include <iostream>
using namespace std;

class Rect;
bool equals(Rect r, Rect s);

class RectManager{
public:
    bool equals(Rect r, Rect s);
};

class Rect{
    int width, height;
public:
    Rect(int width, int height):width(width), height(height){};
    friend RectManager;
};

int main(){
    Rect a(3, 4), b(4, 5);
    RectManager Man;
    if(Man.equals(a, b)) cout << "equal" << endl;
    else cout << "not equal" << endl;

    return 0;
}

bool RectManager::equals(Rect r, Rect s){
    if(r.width == s.width && r.height == s.height) return true;
    else return false;
}
~~~

>> Output

~~~C++
not equal

Process finished with exit code 0
~~~

## Operator Overloading

~~~C++
리턴타입 operator연산자(매개변수)
~~~

+ `C++`에 본래 있는 연산자만 중복 가능
+ 피 연산자 타입이 다른 새로운 연산 정의
+ 연산자는 함수 형태로 구현 - 연산자 함수(Operator function)
  + 클래스의 멤버 함수로 구현
  + 외부 함수로 구현하고 클래스에 프렌드 함수로 선언
+ 반드시 클래스와 관계를 가짐
+ 피연산자의 개수를 바꿀 수 없음
+ 연산의 우선 순위 변경 안됨
+ 모든 연산자가 중복 가능하진 않음

> power_by_member_function.cpp

~~~C++
#include <iostream>
using namespace std;

class Power{
    int kick;
    int punch;
public:
    Power(int kick = 0, int punch = 0):kick(kick), punch(punch){}
    void show();
    Power operator+(Power op2);
};

int main(){
    Power a(3, 5), b(4, 6), c;
    c = a + b; // a.operator+(b)
    a.show();
    b.show();
    c.show();

    return 0;
}

void Power::show(){
    cout << "Kick = " << kick << ',' << " Punch = " << punch << endl;
}
Power Power::operator+(Power op2){
    Power tmp;
    tmp.kick = this->kick + op2.kick;
    tmp.punch = this->punch + op2.punch;
    return tmp;
}
~~~

> power_by_friend_function.cpp

~~~C++
#include <iostream>
using namespace std;

class Power{
    int kick;
    int punch;
public:
    Power(int kick = 0, int punch = 0):kick(kick), punch(punch){}
    void show();
    friend Power operator+(Power op1, Power op2);
};

int main(){
    Power a(3, 5), b(4, 6), c;
    c = a + b; // operator+(a, b)
    a.show();
    b.show();
    c.show();

    return 0;
}

void Power::show(){
    cout << "Kick = " << kick << ',' << " Punch = " << punch << endl;
}
Power operator+(Power op1, Power op2){
    Power tmp;
    tmp.kick = op1.kick + op2.kick;
    tmp.punch = op1.punch + op2.punch;
    return tmp;
}
~~~

>> Output

~~~C++
Kick = 3, Punch = 5
Kick = 4, Punch = 6
Kick = 7, Punch = 11

Process finished with exit code 0
~~~

> cpoint_by_member_function.cpp

~~~C++
#include <iostream>
using namespace std;

class CPoint{
    int x, y;
public:
    CPoint(int a = 0, int b = 0):x(a), y(b){}
    CPoint operator-();
    void Print(){cout << '(' << x << ',' << y << ')' << endl;}
};

int main(){
    CPoint P1(2, 2);
    CPoint P2 = -P1;
    CPoint P3 = -(-P1);

    P1.Print();
    P2.Print();
    P3.Print();

    return 0;
}

CPoint CPoint::operator-(){
    return(CPoint(-this->x, -this->y));
}
~~~

> cpoint_by_friend_function.cpp

~~~C++
#include <iostream>
using namespace std;

class CPoint{
    int x, y;
public:
    CPoint(int a = 0, int b = 0):x(a), y(b){}
    friend CPoint operator-(CPoint obj);
    void Print(){cout << '(' << x << ',' << y << ')' << endl;}
};

int main(){
    CPoint P1(2, 2);
    CPoint P2 = -P1;
    CPoint P3 = -(-P1);

    P1.Print();
    P2.Print();
    P3.Print();

    return 0;
}

CPoint operator-(CPoint obj){
    return CPoint(-obj.x, -obj.y);
}
~~~

>> Output

~~~C++
(2,2)
(-2,-2)
(2,2)

Process finished with exit code 0
~~~

> prefix_by_member_function.cpp

~~~C++
#include <iostream>
using namespace std;

class Power{
    int kick;
    int punch;
public:
    Power(int kick = 0, int punch = 0):kick(kick), punch(punch){}
    void show();
    Power operator++(); // 매개변수 존재 -> postfix
};

int main(){
    Power a(3, 5), b;
    a.show();
    b = ++a;
    a.show();
    b.show();

    return 0;
}

void Power::show(){
    cout << "Kick = " << kick << ',' << " Punch = " << punch << endl;
}
Power Power::operator++(){
    this->kick++;
    this->punch++;
    return *this;
}
~~~

>> Output

~~~C++

~~~

> postfix_by_friend_function.cpp

~~~C++
#include <iostream>
using namespace std;

class Power{
    int kick;
    int punch;
public:
    Power(int kick = 0, int punch = 0):kick(kick), punch(punch){}
    void show();
    friend Power operator++(Power &p, int x); // x 삭제 -> prefix
};

int main(){
    Power a(3, 5), b;
    a.show();
    b = a++;
    a.show();
    b.show();

    return 0;
}

void Power::show(){
    cout << "Kick = " << kick << ',' << " Punch = " << punch << endl;
}
Power operator++(Power &p, int x){
    p.kick++;
    p.punch++;
    return p;
}
~~~

>> Output

~~~C++
Kick = 3, Punch = 5
Kick = 4, Punch = 6
Kick = 4, Punch = 6

Process finished with exit code 0
~~~

> pre_post.cpp

~~~C++
#include <iostream>
using namespace std;

class Power{
    int kick;
    int punch;
public:
    Power(int kick = 0, int punch = 0):kick(kick), punch(punch){}
    void show();
    friend Power operator++(Power &p);
    friend Power operator++(Power &p, int x);
};

int main(){
    Power a(3, 5), b;
    a.show();
    b = ++a;
    a.show();
    b.show();
    Power c(3, 5), d;
    d = c++;
    c.show();
    d.show();

    return 0;
}

void Power::show(){
    cout << "Kick = " << kick << ',' << " Punch = " << punch << endl;
}
Power operator++(Power &p){
    p.kick++;
    p.punch++;
    return p;
}
Power operator++(Power &p, int x){
    p.kick++;
    p.punch++;
    return p;
}
~~~

>> Output

~~~C++
Kick = 3, Punch = 5
Kick = 4, Punch = 6
Kick = 4, Punch = 6
Kick = 4, Punch = 6
Kick = 4, Punch = 6

Process finished with exit code 0
~~~

> complex.cpp

~~~C++
#include <iostream>
using namespace std;

class Complex{
    friend ostream &operator<<(ostream &os, const Complex &v);
    double x, y;
public:
    Complex(double x = 0, double y = 0):x(x), y(y){}
    Complex operator+(const Complex &v2) const{
        Complex v(0.0, 0.0);
        v.x = this->x + v2.x;
        v.y = this->y + v2.y;
        return v;
    }
    void display(){
        cout << '(' << x << ',' << y << 'i' << ')' << endl;
    }
};

int main(){
    Complex v1(1.1,2.1), v2(12.12, 13.13), v3;
    v3 = v1 + v2;
    v1.display();
    v2.display();
    v3.display();
    cout << v1 << v2 << v3;

    return 0;
}

ostream &operator<<(ostream &os, const Complex &v){
    os << '(' << v.x << ',' << v.y << 'i' << ')' << endl;
    return os;
}
~~~

>> Output

~~~C++
(1.1,2.1i)
(12.12,13.13i)
(13.22,15.23i)
(1.1,2.1i)
(12.12,13.13i)
(13.22,15.23i)

Process finished with exit code 0
~~~

## Quiz

> quiz.cpp

~~~C++
#include <iostream>
using namespace std;

class Complex{
    double re, im;
public:
    Complex(double r):re(r), im(0){}
    Complex(double x = 0, double y = 0):re(x),im(y){}
    void Output(){
        cout << re << " + " << im << 'i' << endl;
    }
    Complex &operator+=(Complex com);
    Complex &operator-();
    friend Complex operator+(Complex &com1, Complex &com2);
    friend Complex operator++(Complex &com);
    friend Complex operator++(Complex &com, int x);
    friend ostream &operator<<(ostream &os, Complex &com){
        os << '(' << com.re << ',' << com.im << 'i' << ')' << endl;
        return os;
    }
};

int main(){
    Complex c1(1, 2), c2(3 ,4), c(9, 200);
    c1.Output(); c2.Output(); c1 += c2; c1.Output();
    Complex c3 = c1 + c2;
    Complex c4 = c1 += c2, c5, c6; c3.Output();
    c5 = ++c4; c4.Output(); c5.Output();
    c6 = c4++; c4.Output(); c6.Output();
    c2 = -c2; cout << c2; cout << c;

    return 0;
}

Complex &Complex::operator+=(Complex com){
    this->re = this->re + com.re;
    this->im = this->im + com.im;
    return *this;
}
Complex &Complex::operator-(){
    Complex com(-this->re, -this->im);
    return com;
}
Complex operator+(Complex &com1, Complex &com2){
    Complex com(com1.re + com2.re, com1.im + com2.im);
    return com;
}
Complex operator++(Complex &com){
    com.re++;
    com.im++;
    return com;
}
Complex operator++(Complex &com, int x){
    com.re++;
    com.im++;
    return com;
}
~~~

>> Output

~~~C++
1 + 2i
3 + 4i
4 + 6i
7 + 10i
8 + 11i
8 + 11i
9 + 12i
9 + 12i
(-3,-4i)
(9,200i)

Process finished with exit code 0
~~~

***

# Day 11

## Stack

> stack.cpp

~~~C++
#include <iostream>
using namespace std;

class Stack{
    int size;
    int *mem;
    int tos;
public:
    Stack(int size = 4){
        this->size = size;
        mem = new int[size];
        tos = -1;
    }
    ~Stack(){delete [] mem;}
    Stack &operator<<(int n);
    Stack &operator>>(int &n);
    bool operator!();
};

int main(){
    Stack stack(10);
    stack << 1 << 2 << 3 << 4 << 5;
    while(true){
        if(!stack) break;
        int x;
        stack >> x;
        cout << x << '\t';
    }
    cout << endl;

    return 0;
}

Stack &Stack::operator<<(int n){
    if(tos == size - 1){
        return *this;
    }
    this->tos++;
    this->mem[tos] = n;
    return *this;
}
Stack &Stack::operator>>(int &n){
    if(tos == -1){
        return *this;
    }
    n = this->mem[tos];
    this->tos--;
    return *this;
}
bool Stack::operator!(){
    if(tos == -1)
        return true;
    else
        return false;
}
~~~

>> Output

~~~C++
5	4	3	2	1	

Process finished with exit code 0
~~~

## Const Member & Const Object

+ `const` member variable : 객체 생성과 동시에 초기화 필요
  + 멤버 초기화 구문 사용
+ `const` member function : 멤버 변수의 값을 읽을 수 있으나 변경 불가능
  + 멤버 변수의 주소 반환 불가
  + 비`const` 멤버 함수의 호출 불가
+ `const` object
  + 객체 생성 시 `const` 접두사 추가
  + 멤버 변수의 값 변경 불가
  + `const` 멤버 함수 이외의 멤버 함수에 대한 호출 불가

## Inheritance

+ 기본 클래스(Base class) - 상속해주는 클래스, 부모 클래스
+ 파생 클래스(Derived class) - 상속받는 클래스, 자식 클래스

~~~C++
class Derived : public Base{ //public, private, protected
    ...
}
~~~

+ 간결한 클래스 작성
+ 클래스 간의 계층적 분류 및 관리의 용이함
+ 클래스 재사용과 확장을 통한 소프트웨어 생산성 향상

> inheritance.cpp

~~~C++
#include <iostream>
using namespace std;

class Point{
    int x, y;
public:
    void set(int x, int y){this->x = x; this->y = y;}
    void showPoint(){
        cout << '(' << x << ',' << y << ')' << endl;
    }
};

class ColorPoint : public Point{
    string color;
public:
    void setColor(string color){this->color = color;}
    void showColorPoint();
};

int main(){
    Point p;
    ColorPoint cp;
    cp.set(3, 4);
    cp.setColor("Red");
    cp.showColorPoint();

    return 0;
}

void ColorPoint::showColorPoint(){
    cout << color << " : ";
    showPoint();
}
~~~

>> Output

~~~C++
Red : (3,4)

Process finished with exit code 0
~~~

## Casting

+ 업 캐스팅(Up-casting)
  + 파생 클래스의 객체를 기본 클래스의 포인터로 가리키는 것
  + 포인터 : 기본
  + 객체 : 파생
+ 다운 캐스팅(Down-casting)
  + 기본 클래스 포인터가 가리키는 객체를 파생 클래스의 포인터로 가리키는 것
  + 명시적 형변환 필요
  + 포인터 : 파생
  + 객체 : 기본

## 접근 지정자

+ private 멤버
  + 선언된 클래스 내에서만 접근 가능
  + 파생 클래스에서도 기본 클래스의 private 멤버 직접 접근 불가
+ public 멤버
  + 선언된 클래스나 외부 어떤 클래스, 모든 외부 함수에 접근 허용
  + 파생 클래스에서 기본 클래스의 public 멤버 접근 가능
+ protected 멤버
  + 선언된 클래스에서 접근 가능
  + 파생 클래스에서만 접근 허용

> point.cpp

~~~C++
#include <iostream>
using namespace std;

class Point{
    int x, y;
protected:
    Point(int x, int y):x(x), y(y){}
    int getX(){return x;}
    int getY(){return y;}
    void move(int x, int y){this->x = x; this->y = y;}
};

class ColorPoint : public Point{
    string color;
public:
    ColorPoint():Point(0, 0){color = "BLACK";}
    ColorPoint(int x, int y):Point(x, y){}
    ColorPoint(int x, int y, string color):Point(x, y){this->color = color;}
    void setPoint(int x, int y){move(x, y);}
    void setColor(string color){this->color = color;}
    friend void show(ColorPoint &p);
};

int main(){
    ColorPoint zeroPoint;
    show(zeroPoint);
    ColorPoint cp(5, 5);
    cp.setPoint(10, 20);
    cp.setColor("BLUE");
    show(cp);
    ColorPoint cpRed(23, 33, "RED");
    show(cpRed);

    return 0;
}

void show(ColorPoint &p){
    cout << p.color << "색으로 " << '(' << p.getX() << ',' << p.getY() << ')' << "에 위치한 점입니다." << endl;
}
~~~

>> Output

~~~C++
BLACK색으로 (0,0)에 위치한 점입니다.
BLUE색으로 (10,20)에 위치한 점입니다.
RED색으로 (23,33)에 위치한 점입니다.

Process finished with exit code 0
~~~

## Quiz

> quiz.cpp

~~~C++
#include <iostream>
using namespace std;

class BaseArray{
    int capacity;
    int *mem;
protected:
    BaseArray(int capacity = 100):capacity(capacity){mem = new int[capacity];}
    ~BaseArray(){delete [] mem;}
    void put(int index, int val){mem[index] = val;};
    int get(int index){return mem[index];}
    int getCapacity(){return capacity;};
};

class MyQueue : BaseArray{
    int head;
    int tail;
    int size;
public:
    MyQueue(int capacity):BaseArray(capacity){head = 0; tail = -1; size = 0;}
    void enqueue(int n);
    int dequeue();
    int capacity(){return getCapacity();}
    int length(){return size;}
    void setSize(int S = 0){size = S;}
    int getSize(){return size;}
};

int main(){
    MyQueue mQ(100);
    int Size;
    cout << "큐의 사이즈를 입력하라>> ";
    cin >> Size;
    mQ.setSize(Size);
    int n;
    cout << "큐에 삽입할 "<< mQ.getSize() << "개의 정수를 입력하라>> ";
    for(int i = 0; i < mQ.getSize(); i++){
        cin >> n;
        mQ.enqueue(n);
    }
    cout << "큐의 용량 : " << mQ.capacity() << ",\t큐의 크기 : " << mQ.length() << endl;
    cout << "큐의 원소를 순서대로 제거하여 출력한다>> ";
    while(mQ.length() != 0){
        cout << mQ.dequeue() << ' ';
    }
    cout << endl << "큐의 현재 크기 : " << mQ.length() << endl;

    return 0;
}

void MyQueue::enqueue(int n){
    int he = head % getCapacity();
    put(he, n);
    head++;
}
int MyQueue::dequeue(){
    tail++;
    int ta = tail % getCapacity();
    size--;
    return get(ta);
}
~~~

>> Output

~~~C++
큐의 사이즈를 입력하라>> 5
큐에 삽입할 5개의 정수를 입력하라>> 12 34 44 33 22
큐의 용량 : 100,	큐의 크기 : 5
큐의 원소를 순서대로 제거하여 출력한다>> 12 34 44 33 22 
큐의 현재 크기 : 0

Process finished with exit code 0
~~~