---
title: C++ (2)
date: 2020-07-06 10:10:45
categories:
- Etc.
tags:
- B.S. Course Work
- C, C++
---
# Day 5

## Object

> oval.cpp

~~~C++
#include <iostream>
using namespace std;

class Oval{
    int width, height;
    double getArea();
public:
    Oval();
    Oval(int w, int h);
    ~Oval();
    int getWidth();
    int getHeight();
    void set(int w, int h);
    void show();
};

int main(){
    Oval a, b(3, 4);
    a.set(10, 20);
    a.show();
    b.show();

    return 0;
}

Oval::Oval(){
    width = 1;
    height = 1;
}

Oval::Oval(int w,int h){
    width = w;
    height = h;
}

Oval::~Oval(){
    cout << "Oval 소멸 ";
    show();
}

double Oval::getArea(){
    return 3.14*width*height;
}

int Oval::getWidth(){
    return width;
}

int Oval::getHeight(){
    return height;
}

void Oval::set(int w, int h){
    width = w;
    height = h;
}

void Oval::show(){
    cout << "width = " << width << ", height = " << height << ", Area = " << getArea() << endl;
}
~~~

>> Output

~~~C++
width = 10, height = 20, Area = 628
width = 3, height = 4, Area = 37.68
Oval 소멸 width = 3, height = 4, Area = 37.68
Oval 소멸 width = 10, height = 20, Area = 628

Process finished with exit code 0
~~~

## Object Pointer

+ 객체에 대한 포인터
  + 객체의 주소 값을 가지는 변수
+ 포인터로 멤버를 접근할 때
  + `객체포인터->멤버`

> circle1.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1;}
    Circle(int r){radius = r;}
    double getArea();
};

int main(){
    Circle donut;
    double d = donut.getArea();

    Circle *p;
    p = &donut;
    double b = p->getArea();

    cout << d << "==" << b << endl;

    return 0;
}

double Circle::getArea(){
    return radius*radius*3.14;
}
~~~

>> Output

~~~C++
3.14==3.14

Process finished with exit code 0
~~~

> circle2.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1;}
    Circle(int r){radius = r;}
    double getArea();
};

int main(){
    Circle donut;
    Circle pizza(30);

    cout << donut.getArea() << endl;

    Circle *p;
    p = &donut;
    cout << p->getArea() << endl;
    cout << (*p).getArea() << endl;

    p = & pizza;
    cout << p->getArea() << endl;
    cout << (*p).getArea() << endl;

    return 0;
}

double Circle::getArea(){
    return radius*radius*3.14;
}
~~~

>> Output

~~~C++
3.14
3.14
3.14
2826
2826

Process finished with exit code 0
~~~

## Object Array

+ 객체 배열 선언 가능
  + 기본 타이 배열 선언과 형식 동일
    + `Circle c[3];`
+ 객체 배열 선언
  + 객체 배열을 위한 공간 할당
  + 배열의 각 원소 객체마다 생성자 실행
    + 매개변수 없는 생성자 호출
    + 매개변수 있는 생성자를 한번에는 호출할 수 없음
  + `Circle c[3] = {Circle(10), Circle(20), Circle()};`
+ 배열 소멸
  + 배열의 각 객체마다 소멸자 호출
  + 생성의 반대순으로 소멸

> circle1.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1;}
    Circle(int r){radius = r;}
    double getArea();
    void setRadius(int r){radius = r;}
};

int main(){
    Circle circleArray[3];
    circleArray[0].setRadius(10);
    circleArray[1].setRadius(20);
    circleArray[2].setRadius(30);

    for(int i = 0; i < 3; i++){
        cout << "Circle " << i << "의 면적은 " << circleArray[i].getArea() << endl;
    }

    Circle *p;
    p = circleArray;
    for(int i = 0; i < 3; i++){
        cout << "Circle " << i << "의 면적은 " << p->getArea() << endl;
        p++;
    }

    return 0;
}

double Circle::getArea(){
    return radius*radius*3.14;
}
~~~

>> Output

~~~C++
Circle 0의 면적은 314
Circle 1의 면적은 1256
Circle 2의 면적은 2826
Circle 0의 면적은 314
Circle 1의 면적은 1256
Circle 2의 면적은 2826

Process finished with exit code 0
~~~

> circle2.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1;}
    Circle(int r){radius = r;}
    void setRadius(int r){radius = r;}
    double getArea(){return 3.14*radius*radius;}
};

int main(){
    Circle c[3] = {Circle(10), Circle(20), Circle()};
    Circle *p = c;

    for(int i = 0; i < 3; i++){
        cout << "c[" << i << "]의 면적은 " << c[i].getArea() << endl;
    }

    for(int i = 0; i < 3; i++){
        cout << "(c+" << i << ")의 면적은 " << (c+i)->getArea() << endl;
    }

    for(int i = 0; i < 3; i++){
        cout << "*(c+" << i << ")의 면적은 " << (*(c+i)).getArea() << endl;
    }

    for(int i = 0; i < 3; i++){
        cout << "p[" << i << "]의 면적은 " << p[i].getArea() << endl;
    }

    for(int i = 0; i < 3; i++){
        cout << "(p+" << i << ")의 면적은 " << (p+i)->getArea() << endl;
    }

    for(int i = 0; i < 3; i++){
        cout << "*(p+" << i << ")의 면적은 " << (*(p+i)).getArea() << endl;
    }

    for(int i = 0; i < 3; i++){
        cout << "p->" << i << "의 면적은 " << p->getArea() <<endl;
        p++;
    }

    return 0;
}
~~~

>> Output

~~~C++
c[0]의 면적은 314
c[1]의 면적은 1256
c[2]의 면적은 3.14
(c+0)의 면적은 314
(c+1)의 면적은 1256
(c+2)의 면적은 3.14
*(c+0)의 면적은 314
*(c+1)의 면적은 1256
*(c+2)의 면적은 3.14
p[0]의 면적은 314
p[1]의 면적은 1256
p[2]의 면적은 3.14
(p+0)의 면적은 314
(p+1)의 면적은 1256
(p+2)의 면적은 3.14
*(p+0)의 면적은 314
*(p+1)의 면적은 1256
*(p+2)의 면적은 3.14
p->0의 면적은 314
p->1의 면적은 1256
p->2의 면적은 3.14

Process finished with exit code 0
~~~

## Dynamic Memory

~~~C++
데이터타입 *포인터변수 = new 데이터타입;
데이터타입 *포인터변수 = new 데이터타입(초기값); // 배열의 초기화는 for문
delete 포인터변수;

데이터타입 *포인터변수 = new 데이터타입[배열의크기]; // 배열의 동적 할당
for(int i = 0; i < 배열의크기; i++){
    포인터변수[i] = 값;
}
delete [] 포인터변수;

클래스이름 *포인터변수 = new 클래스이름; // 객체의 동적 할당
클래스이름 *포인터변수 = new 클래스이름(생성자매개변수리스트);
delete 포인터변수;
~~~

+ 정적 할당
  + 변수 선언을 통해 필요한 메모리 할당
  + 많은 양의 메모리는 배열 선언을 통해 할당
+ 동적 할당
  + 필요한 양이 예측되지 않는 경우, 프로그램 작성 시 할당 받을 수 없음
  + 실행 중에 운영체제로부터 할당 받음
    + 힙(`heap`)으로부터 할당
    + 힙은 운영체제가 소유하고 관리하는 메모리, 모든 프로세스가 공유할 수 있는 메모리
+ `C`의 동적 메모리 할당
  + `malloc()`
  + `free()`
+ `C++`의 동적 메모리 할당, 반환
  + `new` 연산자
    + 기본 타입 메모리 할당, 배열 할당, 객체 할당, 객체 배열 할당
    + 객체의 동적 생성 - 힙 메모리로부터 객체를 위한 메모리 할당 요청
    + 객체 할당 시 생성자 호출
  + `delete`
    + `new`로 할당 받은 메모리 반환
    + 객체의 동적 소멸 - 소멸자 호출 뒤 객체를 힙에 반환

|Data Type|정적 할당|동적 할당|
|:--:|:--:|:--:|
|Integer|`int a = val; int *pa = &a;`|`int *p = new int(val);`|
|Array|`int arr[num];`|`int *parr = new int[100];`|
|Object|`Class instance(val); Class *pc = &instance;`|`Class *pc = new Class;`|

> circle.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1; cout << "생성자 실행" << endl;}
    Circle(int r){radius = r; cout << "생성자 실행" << endl;}
    ~Circle(){cout << "소멸자 실행" << endl;}
    double getArea(){return 3.14*radius*radius;}
};

int main(){
    int radius;
    while(true){
        cout << "정수 반지름 입력(음수이면 종료 >> ";
        cin >> radius;
        if(radius < 0) break;
        Circle *p = new Circle(radius);
        cout << "원의 면적은 " << p->getArea() << endl;
        delete p;
    }
    return 0;
}
~~~

>> Output

~~~C++
정수 반지름 입력(음수이면 종료 >> 5
생성자 실행
원의 면적은 78.5
소멸자 실행
정수 반지름 입력(음수이면 종료 >> 9
생성자 실행
원의 면적은 254.34
소멸자 실행
정수 반지름 입력(음수이면 종료 >> -1

Process finished with exit code 0
~~~

## This Pointer

+ 현재의 실행중인 객체를 가리키는 `pointer` 변수
+ 포인터, 객체 자신 포인터
+ 클래스의 멤버 함수 내에서만 사용
+ 개발자가 선언하는 변수가 아닌 컴파일러가 선언한 변수
+ 용도
  + 매개변수 이름 == 멤버 변수 이름
  + 매개변수가 자신의 객체주소를 `return`
+ 사용범위
  + 멤버 함수

## Quiz

> quiz1.cpp

~~~C++
#include <iostream>
using namespace std;

class Sample{
    int *p;
    int size;
public:
    Sample(int n){
        size = n; p = new int[n];
    }
    void read();
    void write();
    int big();
    int getSize(){return size;}
    ~Sample(){delete [] p;}
};

int main(){
    Sample s(10);
    s.read();
    cout << "동적배열 정수 " << s.getSize() << "개를 출력합니다. ";
    s.write();
    cout << "가장 큰 수는 " << s.big() << endl;

    return 0;
}

void Sample::read(){
    cout << "입력하려는 정수의 개수는? ";
    cin >> size;
    cout << size << "개의 정수를 입력하시오. ";
    for(int i = 0; i < size; i++)
        cin >> p[i];
}
void Sample::write(){
    for(int i = 0; i < size; i++)
        cout << p[i] << ' ';
    cout << endl;
}
int Sample::big(){
    int b = p[0];
    for(int i = 0; i < size; i++){
        b = (b < p[i]) ? p[i] : b;
    }
    return b;
}
~~~

>> Output

~~~C++
입력하려는 정수의 개수는? 5
5개의 정수를 입력하시오. 11 22 44 55 23
동적배열 정수 5개를 출력합니다. 11 22 44 55 23 
가장 큰 수는 55

Process finished with exit code 0
~~~

> quiz2.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1;}
    Circle(int r){radius = 1;}
    void setRadius(int r){radius = r;}
    int getRadius(){return radius;}
    double getArea(){return 3.14 * radius * radius;}
};

class Sample{
    Circle *p;
    int size;
public:
    Sample(int n){
        size = n; p = new Circle[n];
    }
    void read();
    void write();
    Circle big();
    int getSize(){return size;}
    ~Sample(){delete [] p;}
};

int main(){
    Sample s(10);
    s.read();
    cout << "각 원 객체의 반지름 " << s.getSize() << "개를 출력합니다. ";
    s.write();
    Circle big = s.big();
    cout << "가장 큰 원의 넓이 : " << big.getArea() << "\t 가장 큰 원의 반지름 : " << big.getRadius() << endl;

    return 0;
}

void Sample::read(){
    int num;
    cout << "입력하려는 원의 개수는? ";
    cin >> size;
    cout << size << "개의 원의 반지름을 입력하시오. ";
    for(int i = 0; i < size; i++){
        cin >> num;
        p[i].setRadius(num);
    }
}
void Sample::write(){
    for(int i = 0; i < size; i++)
        cout << p[i].getRadius() << ' ';
    cout << endl;
}
Circle Sample::big(){
    Circle b = p[0];
    for(int i = 0; i < size; i++){
        b.setRadius((b.getRadius() < p[i].getRadius()) ? p[i].getRadius() : b.getRadius());
    }
    return b;
}
~~~

>> Output

~~~C++
입력하려는 원의 개수는? 3
3개의 원의 반지름을 입력하시오. 3 16 2
각 원 객체의 반지름 3개를 출력합니다. 3 16 2 
가장 큰 원의 넓이 : 803.84	 가장 큰 원의 반지름 : 16

Process finished with exit code 0
~~~

***

# Day 6

## string

+ 문자열 생성
  + `string str0("name");`
  + `string str1 = "name";`
  + `string str2(str);` - 복사생성자
  + 주의 : 문자열 끝에 `'\0(NULL)`가 없음
+ 문자열 연산자
  + 산술 연산자
    + `+=`, `+`
  + 관계 연산자
    + `>=`, `<=`, `>`, `<`, `!=`, `==`
    + 비교는 사전식
  + 배열처럼 사용 가능
+ 문자열 변환 함수
  + 문자열 -> 숫자
    + `stoi();`
    + `stof();`
    + `stod();`
  + 숫자 -> 문자열
    + `to_string();`
  + 문자열 -> C언어 문자열(`\0`)
    + `str.c_str();`
+ 문자열 크기 함수
  + `str.size();`
  + `str.length();`
  + `str.capacity();` - 시스템이 정해줌
+ 문자열 조작 함수
  + `str.append(string);`
    + 문자열 뒤에 파라미터의 문자열 추가
  + `str.substr(시작 index, 크기);`
    + 시작부터 크기만큼 추출
  + `str.replace(index, length, string);`
    + `index`에서 `length`만큼 `string`으로 대체
  + `str.find(string, index);`
    + `index`부터 `string`이 시작하는 위치 `int`값으로 `return`
  + `str.resize(unsigned, char);`
    + 숫자만큼의 크기로 바꾸며 남는다면 `char`으로 채움

> string1.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    string str;
    string address("서울시 서울시 서울시");
    string copyAddress(address);

    char text[] = {'L', 'O', 'V', 'E', ' ', 'C', '+', '+', '\0'};
    string title(text);

    cout << str << endl;
    cout << address << endl;
    cout << copyAddress << endl;
    cout << title << endl;

    return 0;
}
~~~

>> Output

~~~C+

서울시 서울시 서울시
서울시 서울시 서울시
LOVE C++

Process finished with exit code 0
~~~

> string2.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    string str;

    cout << "문자열을 입력하세요 : ";

    getline(cin, str, '\n');
    int len = str.length();

    for(int i = 0; i < len; i++){
        string first = str.substr(0, 1);
        string sub = str.substr(1, len-1);
        str = sub + first;
        cout << str <<endl;
    }

    return 0;
}
~~~

>> Output

~~~C++
문자열을 입력하세요 : asdfasdfasdfasdf
sdfasdfasdfasdfa
dfasdfasdfasdfas
fasdfasdfasdfasd
asdfasdfasdfasdf
sdfasdfasdfasdfa
dfasdfasdfasdfas
fasdfasdfasdfasd
asdfasdfasdfasdf
sdfasdfasdfasdfa
dfasdfasdfasdfas
fasdfasdfasdfasd
asdfasdfasdfasdf
sdfasdfasdfasdfa
dfasdfasdfasdfas
fasdfasdfasdfasd
asdfasdfasdfasdf

Process finished with exit code 0
~~~

> string3.cpp

~~~C++
#include <iostream>
using namespace std;

class Date{
    string year;
    string month;
    string day;
public:
    Date(int y, int m, int d){year = to_string(y); month = to_string(m); day = to_string(d);}
    Date(string when);
    void show();
    string getYear(){return year;}
    string getMonth(){return month;}
    string getDay(){return day;}
};

int main(){
    Date birth(2014, 3, 20);
    Date independenceDay("1945/8/15");
    independenceDay.show();
    birth.show();
    cout << birth.getYear() << ',' << birth.getMonth() << ',' << birth.getDay() << endl;

    return 0;
}

Date::Date(string when){
    int where1;
    int where2;
    where1 = when.find('/');
    year = when.substr(0, where1);
    where2 = when.find('/', where1 + 1);
    month = when.substr(where1 + 1, where2 - where1 - 1);
    day = when.substr(where2 + 1, when.size());
}
void Date::show(){
    cout << year << "년" << month << "월" << day << "일" << endl;
}
~~~

>> Output

~~~C++
1945년8월15일
2014년3월20일
2014,3,20

Process finished with exit code 0
~~~

## Call by Reference

~~~C++
int n = 0;
int &refn = n;

class c;
class &refc = c;
~~~

+ 참조 변수
  + 참조자 `&`의 도입
  + 이미 존재하는 변수에 대한 다른 이름(별명)을 선언
    + 참조 변수는 이름만 존재
    + 참조 변수에 새로운 공간 할당 X
    + 초기화로 지정된 기존 변수 공유

> reference1.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    cout << 'i' << '\t' << 'n' << '\t' << "refn" << endl;

    int i = 1;
    int n = 2;
    int &refn = n;
    n = 4;
    refn++;
    cout << i << '\t' << n << '\t' << refn << endl;

    refn = i;
    refn++;
    cout << i << '\t' << n << '\t' << refn << endl;

    int *p = &refn;
    *p = 20;
    cout << i << '\t' << n << '\t' << refn << endl;

    return 0;
}
~~~

>> Output

~~~C++
i	n	refn
1	5	5
1	2	2
1	20	20

Process finished with exit code 0
~~~

> reference2.cpp

~~~C++
#include <iostream>
using namespace std;

class Circle{
    int radius;
public:
    Circle(){radius = 1;}
    Circle(int radius){this->radius = radius;}
    void setRadius(int radius){this->radius = radius;}
    double getArea(){return 3.14*radius*radius;}
};

void readRadius(Circle &c);

int main(){
    Circle donut;
    readRadius(donut);
    cout << "donut의 면적 = " << donut.getArea() << endl;

    return 0;
}

void readRadius(Circle &c){
    int r;
    cout << "정수 값으로 반지름을 입력하세요 >> ";
    cin >> r;
    c.setRadius(r);
}
~~~

>> Output

~~~C++
정수 값으로 반지름을 입력하세요 >> 3
donut의 면적 = 28.26

Process finished with exit code 0
~~~

> reference3.cpp

~~~C++
#include <iostream>
using namespace std;

class MyintStack{
    int p[10];
    int tos;
public:
    MyintStack();
    bool push(int n);
    bool pop(int &refn);
};

int main(){
    MyintStack a;
    for(int i = 0; i < 11; i++){
        if(a.push(i))
            cout << i << '\t';
        else
            cout << endl << i + 1 << "번째 stack full" << endl;
    }
    int n;
    for(int i = 0; i < 11; i++){
        if(a.pop(n))
            cout << n << '\t';
        else
            cout << endl << i + 1 << "번째 stack empty" << endl;
    }

    return 0;
}

MyintStack::MyintStack(){
    tos = 0;
}
bool MyintStack::push(int n){
    if(tos == 10)
        return false;
    p[tos] = n;
    tos++;
    return true;
}
bool MyintStack::pop(int &refn){
    if(tos == 0)
        return false;
    tos--;
    refn = p[tos];
    return true;
}
~~~

>> Output

~~~C++
0	1	2	3	4	5	6	7	8	9	
11번째 stack full
9	8	7	6	5	4	3	2	1	0	
11번째 stack empty

Process finished with exit code 0
~~~

## Quiz

> quiz.cpp

~~~C++
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
using namespace std;

class Player{
    string name;
public:
    Player(string name = ""){
        this->name = name;
    }
    void setName(string name){this->name = name;}
    string getName(){return name;}
    void getEnterKey(){
        char buf[100];
        cin.getline(buf, 99);
    }
};

class GamblingGame{
    Player p[2];
    int num[3];
    bool matchAll();
public:
    GamblingGame();
    void run();
};

int main(){
    GamblingGame game;
    game.run();

    return 0;
}

GamblingGame::GamblingGame(){
    cout << "***** 갬블링 게임을 시작합니다. *****" << endl;
    srand((unsigned int) time(0));
    for(int i = 0; i < 3; i++)
        num[i] = 0;
    cout << "첫번째 선수 이름>>";
    string name;
    cin >> name;
    p[0].setName(name);
    cout << "두번째 선수 이름>>";
    cin >> name;
    p[1].setName(name);
}
void GamblingGame::run(){
    string n;
    int i = 0;
    for(;;){
        cout << p[i % 2].getName() << " : <Enter>";
        if(i == 0)
            cout << endl;
        p[i % 2].getEnterKey();
        if(this->matchAll()){
            cout << p[i % 2].getName() << "님 승리!!" << endl;
            break;
        }
        else
            cout << "아쉽군요!" << endl;
        i++;
    }
}
bool GamblingGame::matchAll(){
    cout << "\t";
    for (int i = 0; i < 3; i++){
        int n = rand() % 3;
        num[i] = n;
        cout << num[i] << "\t";
    }
    if (num[0] == num[1] && num[0] == num[2])
        return true;
    else
        return false;
}
~~~

>> Output

~~~C++
***** 갬블링 게임을 시작합니다. *****
첫번째 선수 이름>>Kim
두번째 선수 이름>>Park
Kim : <Enter>
	0	1	1	아쉽군요!
Park : <Enter>
	2	0	2	아쉽군요!
Kim : <Enter>
	1	2	1	아쉽군요!
Park : <Enter>
	1	1	1	Park님 승리!!

Process finished with exit code 0
~~~

***

# Day 7

## Dynamic Memory

> stack.cpp

~~~C++
#include <iostream>
using namespace std;

class MyintStack{
    int *parr;
    int tos;
public:
    MyintStack(int size);
    bool push(int n);
    bool pop(int &refn);
    ~MyintStack(){delete [] parr;}
};

int main(){
    MyintStack a(10);
    for(int i = 0; i < 11; i++){
        if(a.push(i))
            cout << i << '\t';
        else
            cout << endl << i + 1 << "번째 stack full" << endl;
    }
    int n;
    for(int i = 0; i < 11; i++){
        if(a.pop(n))
            cout << n << '\t';
        else
            cout << endl << i + 1 << "번째 stack empty" << endl;
    }

    return 0;
}

MyintStack::MyintStack(int size){
    tos = 0;
    parr = new int[size];
}
bool MyintStack::push(int n){
    if(tos == 10)
        return false;
    parr[tos] = n;
    tos++;
    return true;
}
bool MyintStack::pop(int &refn){
    if(tos == 0)
        return false;
    tos--;
    refn = parr[tos];
    return true;
}
~~~

>> Output

~~~C++
0	1	2	3	4	5	6	7	8	9	
11번째 stack full
9	8	7	6	5	4	3	2	1	0	
11번째 stack empty

Process finished with exit code 0
~~~

## Return Reference

> return_ref1.cpp

~~~C++
#include <iostream>
using namespace std;

char c = 'a';

char &find(){
    return c;
}

int main(){
    char a = find();
    cout << a << endl;

    char &ref = find();
    cout << ref << endl;

    find() = 'b';
    cout << find() << endl;

    return 0;
}
~~~

>> Output

~~~C++
a
a
b

Process finished with exit code 0
~~~

> return_ref2.cpp

~~~C++
#include <iostream>
using namespace std;

char &find(char s[], int index){
    return s[index];
}

int main(){
    char name[] = "Zerohertz";
    cout << name << endl;

    find(name, 0) = '5';
    cout << name << endl;

    char &ref = find(name, 2);
    ref = 't';
    cout << name << endl;

    return 0;
}
~~~

>> Output

~~~C++
Zerohertz
5erohertz
5etohertz

Process finished with exit code 0
~~~

> reference.cpp

~~~C++
#include <iostream>
using namespace std;

void addConst(int &x, int y){
    x = x + 200;
    y = y + 200;
    cout << "addConst" << endl;
    cout << "&x = " << &x << "\tx = " << x << endl;
    cout << "&y = " << &y << "\ty = " << y << endl;
}

int main(){
    int a = 100;
    int b = 100;
    addConst(a, b);
    cout << "Main" << endl;
    cout << "&a = " << &a << "\ta = " << a << endl;
    cout << "&b = " << &b << "\tb = " << b << endl;

    return 0;
}
~~~

>> Output

~~~C++
addConst
&x = 0x7ffee84d09a8	x = 300
&y = 0x7ffee84d0964	y = 300
Main
&a = 0x7ffee84d09a8	a = 300
&b = 0x7ffee84d09a4	b = 100

Process finished with exit code 0
~~~

> return_ref3.cpp

~~~C++
#include <iostream>
using namespace std;

int &addConst(int &x, int y){
    x = x + 200;
    y = y + 200;
    cout << "addConst" << endl;
    cout << "&x = " << &x << "\tx = " << x << endl;
    cout << "&y = " << &y << "\ty = " << y << endl;
    return x;
}

int main(){
    int a = 100;
    int b = 100;
    addConst(a, b) = 555;
    cout << "Main" << endl;
    cout << "&a = " << &a << "\ta = " << a << endl;
    cout << "&b = " << &b << "\tb = " << b << endl;

    return 0;
}
~~~

>> Output

~~~C++
addConst
&x = 0x7ffee834b9a8	x = 300
&y = 0x7ffee834b964	y = 300
Main
&a = 0x7ffee834b9a8	a = 555
&b = 0x7ffee834b9a4	b = 100

Process finished with exit code 0
~~~

## Shallow Copy & Deep Copy

+ Shallow copy
  + 객체 복사 시, 객체의 멤버를 1:1로 복사
  + 객체의 멤버 변수에 동적 메모리가 할당된 경우
    + 사본은 원본 객체가 할당 받은 메모리를 공유하는 문제 발생
+ Deep copy
  + 객체 복사 시, 객체의 멤버를 1:1로 복사
  + 객체의 멤버 변수에 동적 메모리가 할당된 경우
    + 사본은 원본이 가진 메모리 크기만큼 별도로 동적 할당
    + 원본의 동적 메모리에 있는 내용을 사본에 복사
  + 완전한 형태의 복사
    + 사본과 원본은 메모리를 공유하는데 문제 없음

> shallow1_copy.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int *a = new int(3);
    int *b = new int(5);
    cout << "a의 주소(복사전) : " << a << endl;
    cout << "b의 주소(복사전) : " << b << endl;

    a = b;

    cout << "a의 주소(복사후) : " << a << endl;
    cout << "b의 주소(복사후) : " << b << endl;

    cout << "a의 값 : " << *a << endl;
    cout << "b의 값 : " << *b << endl;

    delete a;
    delete b;

    return 0;
}
~~~

>> Output

~~~C++
untitled(997,0x11451edc0) malloc: *** error for object 0x7fefdc405980: pointer being freed was not allocated
untitled(997,0x11451edc0) malloc: *** set a breakpoint in malloc_error_break to debug
a의 주소(복사전) : 0x7fefdc405970
b의 주소(복사전) : 0x7fefdc405980
a의 주소(복사후) : 0x7fefdc405980
b의 주소(복사후) : 0x7fefdc405980
a의 값 : 5
b의 값 : 5

Process finished with exit code 6
~~~

> shallow2_copy.cpp

~~~C++
#include <iostream>
#include <cstring>
#pragma warning(disable:4996)
using namespace std;

class Person{
    char *name;
    int id;
public:
    Person(Person &p){
        this->name = p.name;
        this->id = p.id;
    }
    Person(int id, const char *name);
    ~Person();
    void changeName(const char *name);
    void show(){cout << id << ',' << name << endl;}
};

int main(){
    Person zerohertz(1, "zerohertz");
    Person zhz(zerohertz);

    cout << "***** zhz 객체 생성 후 *****" << endl;
    zerohertz.show();
    zhz.show();

    zhz.changeName("0Hz");
    cout << "***** zhz 이름 변경 후 *****" << endl;
    zerohertz.show();
    zhz.show();

    return 0; // zhz, zerohertz 순으로 소멸, zerohertz 소멸 시 오류
}

Person::Person(int id, const char *name){
    this->id = id;
    int len = strlen(name);
    this->name = new char[len + 1];
    strcpy(this->name, name);
}
Person::~Person(){
    delete [] name;
}
void Person::changeName(const char *name){
    if(strlen(name) > strlen(this->name))
        return;
    strcpy(this->name, name);
}
~~~

>> Output

~~~C++
untitled(1347,0x11c7dddc0) malloc: *** error for object 0x7f9184c05970: pointer being freed was not allocated
untitled(1347,0x11c7dddc0) malloc: *** set a breakpoint in malloc_error_break to debug
***** zhz 객체 생성 후 *****
1,zerohertz
1,zerohertz
***** zhz 이름 변경 후 *****
1,0Hz
1,0Hz

Process finished with exit code 6
~~~

## Copy Constructor

> copy_constructor1.cpp

~~~C++
#include <iostream>
#include <cstring>
#pragma warning(disable:4996)
using namespace std;

class Person{
    char *name;
    int id;
public:
    Person(Person &p);
    Person(int id, const char *name);
    ~Person();
    void changeName(const char *name);
    void show(){cout << id << ',' << name << endl;}
};

int main(){
    Person zerohertz(1, "zerohertz");
    Person zhz(zerohertz);

    cout << "***** zhz 객체 생성 후 *****" << endl;
    zerohertz.show();
    zhz.show();

    zhz.changeName("0Hz");
    cout << "***** zhz 이름 변경 후 *****" << endl;
    zerohertz.show();
    zhz.show();

    return 0;
}

Person::Person(Person &p){
    this->id = p.id;
    int len = strlen(p.name);
    this->name = new char[len + 1];
    strcpy(this->name, p.name);
    cout << "복사 생성자 실행, 원본 객체의 이름 : " << this->name << endl;
}
Person::Person(int id, const char *name){
    this->id = id;
    int len = strlen(name);
    this->name = new char[len + 1];
    strcpy(this->name, name);
}
Person::~Person(){
    delete [] name;
}
void Person::changeName(const char *name){
    if(strlen(name) > strlen(this->name))
        return;
    strcpy(this->name, name);
}
~~~

>> Output

~~~C++
복사 생성자 실행, 원본 객체의 이름 : zerohertz
***** zhz 객체 생성 후 *****
1,zerohertz
1,zerohertz
***** zhz 이름 변경 후 *****
1,zerohertz
1,0Hz

Process finished with exit code 0
~~~

> copy_constructor2.cpp

~~~C++
#include <iostream>
#pragma warning(disable:4996)
using namespace std;

class MyString{
    char *pBuf;
public:
    MyString(const char *s = NULL);
    MyString(MyString &MyStr);
    ~MyString();
    void print();
    int getSize();
};

int main(){
    MyString str1;
    MyString str2("Hello");
    MyString str3("World!");
    MyString str4(str3);
    str1.print();
    str2.print();
    str3.print();
    str4.print();

    return 0;
}

MyString::MyString(const char *s){
    if(s == NULL){
        pBuf = new char[1];
        pBuf[0] = NULL;
    }
    else{
        pBuf = new char[strlen(s) + 1];
        strcpy(pBuf, s);
    }
}
MyString::MyString(MyString &MyStr){
    int len = MyStr.getSize();
    this->pBuf = new char[len + 1];
    strcpy(this->pBuf, MyStr.pBuf);
}
MyString::~MyString(){
    if(pBuf) delete [] pBuf;
}
void MyString::print(){
    cout << pBuf << endl;
}
int MyString::getSize(){
    return strlen(pBuf);
}
~~~

>> Output

~~~C++

Hello
World!
World!

Process finished with exit code 0
~~~

**객체에서 동적 메모리 사용 시 복사 생성자 직접 생성**