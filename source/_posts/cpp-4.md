---
title: C++ (4)
date: 2020-07-06 10:10:47
categories:
- Etc.
tags:
- B.S. Course Work
- C, C++
---
# Day 12

## Constructor of Inheritance

> constructor.cpp

~~~C++
#include <iostream>
using namespace std;

class TV{
    int size;
public:
    TV(){size = 20;}
    TV(int size):size(size){}
    int getSize(){return size;}
};

class WideTV : public TV{
    bool videoIn;
public:
    WideTV(int size, bool videoIn) : TV(size){
        this->videoIn = videoIn;
    }
    bool getVideoIn(){return videoIn;}
};

class SmartTV : public WideTV{
    string ipAddr;
public:
    SmartTV(string ipAddr, int size) : WideTV(size, true){
        this->ipAddr = ipAddr;
    }
    string getipAddr(){return ipAddr;}
};

int main(){
    SmartTV htv("192.0.0.1", 32);
    cout << "size = " << htv.getSize() << endl;
    cout << "videoIn = " << htv.getVideoIn() << endl;
    cout << "IP = " << htv.getipAddr() << endl;

    return 0;
}
~~~

>> Output

~~~C++
size = 32
videoIn = 1
IP = 192.0.0.1

Process finished with exit code 0
~~~

## Virtual Function & Overriding

+ Virtual function
  + `virtual` 키워드로 선언된 멤버 함수
  + 동적 바인딩 지시어
  + 컴파일러에게 함수에 대한 호출 바인딩을 실행 시간까지 미루도록 지시
+ Function overriding
  + 파생 클래스에서 기본 클래스의 가상 함수와 동일한 이름의 함수 선언
  + 기본 클래스 : 가상 함수의 존재감 상실
  + 파생 클래스 : 오버라이딩한 함수가 호출되도록 동적 바인딩
  + 함수 재정의라고도 부름
  + 다형성의 한 종류
+ 조건
  + `virtual`으로 함수 선언(파생 클래스는 생략 가능)
  + upcasting
  + 함수 동일

> overriding.cpp

~~~C++
#include <iostream>
using namespace std;

class Base{
public:
    virtual void f(){cout << "Base" << endl;}
};

class Derived : public Base{
public:
    void f(){cout << "Derived" << endl;}
};

class GrandDerived : public Derived{
public:
    void f(){cout << "GrandDerived" << endl;}
};

int main(){
    Base *bp = new GrandDerived;
    bp->f();
    Derived *dp = new GrandDerived;
    dp->f();

    return 0;
}
~~~

>> Output

~~~C++
GrandDerived
GrandDerived

Process finished with exit code 0
~~~

> destructor1.cpp

~~~C++
#include <iostream>
using namespace std;

class Base{
public:
    ~Base(){cout << "~Base" << endl;}
};

class Derived : public Base{
public:
    ~Derived(){cout << "~Derived" << endl;}
};

class GrandDerived : public Derived{
public:
    ~GrandDerived(){cout << "~GrandDerived" << endl;}
};

int main(){
    Base *bp = new GrandDerived;
    Derived *dp = new GrandDerived;
    GrandDerived *gp = new GrandDerived;

    delete bp;
    delete dp;
    delete gp;

    return 0;
}
~~~

>> Output

~~~C++
~Base
~Derived
~Base
~GrandDerived
~Derived
~Base

Process finished with exit code 0
~~~

> destructor2.cpp

~~~C++
#include <iostream>
using namespace std;

class Base{
public:
    virtual ~Base(){cout << "~Base" << endl;}
};

class Derived : public Base{
public:
    ~Derived(){cout << "~Derived" << endl;}
};

class GrandDerived : public Derived{
public:
    ~GrandDerived(){cout << "~GrandDerived" << endl;}
};

int main(){
    Base *bp = new GrandDerived;
    Derived *dp = new GrandDerived;
    GrandDerived *gp = new GrandDerived;

    delete bp;
    delete dp;
    delete gp;

    return 0;
}
~~~

>> Output

~~~C++
~GrandDerived
~Derived
~Base
~GrandDerived
~Derived
~Base
~GrandDerived
~Derived
~Base

Process finished with exit code 0
~~~

## Overloading vs. Overrding

+ Overloading
  + 이름만 같은 함수 중복 작성
  + 하나의 클래스
+ Overriding
  + 모든 것이 완벽히 같은 함수 재작성
  + 상속

## Quiz

> quiz.cpp

~~~C++
#include <iostream>
using namespace std;

class BaseArray{
    int capacity;
    int *mem;
public:
    BaseArray(int capacity = 100):capacity(capacity){mem = new int[capacity];}
    ~BaseArray(){delete [] mem;}
    void put(int index, int val);
    int get(int index);
    int getCapacity(){return capacity;}
};

class MyStack : public BaseArray{
    int tos;
public:
    MyStack(int capacity) : BaseArray(capacity){tos = 0;}
    void push(int n);
    int pop();
    int capacity(){return getCapacity();}
    int length(){return tos;}
};

int main(){
    MyStack mStack(100);
    int n;
    cout << "스택에 삽입할 5개의 정수를 입력하라>> ";
    for(int i = 0; i < 5; i++){
        cin >> n;
        mStack.push(n);
    }
    cout << "스택 용량:" << mStack.capacity() << ", 스택 크기:" << mStack.length() << endl;
    cout << "스택의 모든 원소를 팝하여 출력한다>> ";
    while(mStack.length() != 0){
        cout << mStack.pop() << ' ';
    }
    cout << endl << "스택의 현재 크기 : " << mStack.length() << endl;
}

void BaseArray::put(int index, int val){
    mem[index] = val;
}
int BaseArray::get(int index){
    return mem[index];
}

void MyStack::push(int n){
    put(tos, n);
    tos++;
}
int MyStack::pop(){
    tos--;
    return get(tos);
}
~~~

>> Output

~~~C++
스택에 삽입할 5개의 정수를 입력하라>> 34 52 41 12 78
스택 용량:100, 스택 크기:5
스택의 모든 원소를 팝하여 출력한다>> 78 12 41 52 34 
스택의 현재 크기 : 0

Process finished with exit code 0
~~~

***

# Day 13

## Interface

+ 인터페이스만 선언하고 구현을 분리하여 작업자마다 다양한 구현 가능
+ 사용자는 구현의 내용을 모르지만 인터페이스에 선언된 순수 가상 함수가 구현되어있기 때문에 호출하여 사용하기만 하면 됨

> shape.cpp

~~~C++
#include <iostream>
using namespace std;

class Shape{
    Shape *next;
protected:
    virtual void draw() = 0; // 순수 가상 함수
public:
    Shape(){next = NULL;}
    virtual ~Shape(){}
    void paint(){draw();}
    Shape *add(Shape *p){this->next = p; return p;};
    Shape *getNext(){return next;}
};

class Circle : public Shape{
protected:
    void draw(){cout << "Circle" << endl;}
    ~Circle(){cout << "del Circle" << endl;}
};

class Rect : public Shape{
protected:
    void draw(){cout << "Rect" << endl;}
    ~Rect(){cout << "del Rect" << endl;}
};

int main(){
    Shape *pStart = NULL;
    Shape *pLast;
    pStart = new Circle();
    pLast = pStart;
    pLast = pLast->add(new Rect());
    pLast = pLast->add(new Rect());
    pLast = pLast->add(new Circle());
    pLast = pLast->add(new Rect());
    Shape *p = pStart;
    while(p != NULL){
        p->paint();
        p = p->getNext();
    }
    p = pStart;
    while(p != NULL){
        Shape *q = p->getNext();
        delete p;
        p = q;
    }

    return 0;
}
~~~

>> Output

~~~C++
Circle
Rect
Rect
Circle
Rect
del Circle
del Rect
del Rect
del Circle
del Rect

Process finished with exit code 0
~~~

## Abstract Class

+ 최소한 하나의 순수 가상 함수를 가진 클래스
+ 온전한 클래스가 아니므로 객체 생성 불가능
+ 추상 클래스의 포인터는 선언 가능
+ 순수 가상 함수를 통해 파생 클래스에서 구현할 함수의 형태(원형)을 보여주는 인터페이스 역할

> calculator.cpp

~~~C++
#include <iostream>
using namespace std;

class Calculator{
    void input(){
        cout << "정수 2개를 입력하세요>>";
        cin >> a >> b;
    }
protected:
    int a, b;
    virtual int calc(int a, int b) = 0;
public:
    void run(){
        input();
        cout << "계산된 값은 " << calc(a, b) << endl;
    }
};

class Adder : public Calculator{
protected:
    int calc(int a, int b){return a + b;}
};

class Subtract : public Calculator{
protected:
    int calc(int a, int b){return a - b;}
};

int main(){
    Calculator *c;
    c = new Adder;
    c->run();
    c = new Subtract;
    c->run();
    delete c;

    return 0;
}
~~~

>> Output

~~~C++
정수 2개를 입력하세요>>4 3
계산된 값은 7
정수 2개를 입력하세요>>4 3
계산된 값은 1

Process finished with exit code 0
~~~

## Generalization of Function

+ Generic 혹은 일반화
  + 함수나 클래스를 일반화시키고, 매개변수 타입을 지정하여 틀에서 찍어내듯이 함수나 클래스 코드를 생산하는 기법
+ Template
  + 함수나 클래스를 일반화하는 `C++` 도구
  + `template` 키워드로 함수나 클래스 선언
  + Generic type - 일반화를 위한 Data type

> sum.cpp

~~~C++
#include <iostream>
using namespace std;

template<class T>
T Sum(T a, T b){
    return a + b;
}

int main(){
    cout << Sum(1, 2) << endl;
    cout << Sum(1.1, 2.2) << endl;
    cout << Sum('1', '2') << endl;

    return 0;
}
~~~

>> Output

~~~C++
3
3.3
c

Process finished with exit code 0
~~~

> search.cpp

~~~C++
#include <iostream>
using namespace std;

template<class T>
bool search(T one, T arr[], int size){
    for(int i = 0; i < size; i++){
        if(arr[i] == one)
            return true;
    }
    return false;
}

int main(){
    int x[] = {1, 10, 100, 5, 4};
    if(search(100, x, sizeof(x) / 4))
        cout << "100이 배열 x에 포함되어 있다.";
    else
        cout << "100이 배열 x에 포함되어 있지 않다.";
    cout << endl;

    char c[] = {'h', 'e', 'l', 'l', 'o'};
    if(search('e', c, 5))
        cout << "e가 배열 x에 포함되어 있다.";
    else
        cout << "e가 배열 x에 포함되어 있지 않다.";
    cout << endl;

    return 0;
}
~~~

>> Output

~~~C++
100이 배열 x에 포함되어 있다.
e가 배열 x에 포함되어 있다.

Process finished with exit code 0
~~~

## Generalization of Class

+ 선언 : `template<class T>`
  + class의 정의 앞에 선언
  + 선언부, 구현부 - 멤버 함수 앞 선언
  + `T class<T>::function(T param);`
+ 일반화할 변수만 `T`로 선언

> generic.cpp

~~~C++
#include <iostream>
using namespace std;

class Point{
    int x, y;
public:
    Point(int x = 0, int y = 0):x(x), y(y){}
    void show(){cout << '(' << x << ',' << y << ')' << endl;}
};

template<class T>
class MyStack{
    int tos;
    T data[100];
public:
    MyStack();
    void push(T element);
    T pop();
};

int main(){
    MyStack<int *> ipStack;
    int *p = new int[3];
    for(int i = 0; i < 3; i++)
        p[i] = i * 10;
    ipStack.push(p);
    int *q = ipStack.pop();
    for(int i = 0; i < 3; i++)
        cout << q[i] << '\t';
    cout << endl;
    delete [] p;

    MyStack<Point> pointStack;
    Point a(2, 3), b;
    pointStack.push(a);
    b = pointStack.pop();
    b.show();

    MyStack<Point *> pStack;
    pStack.push(new Point(10, 20));
    Point *pPoint = pStack.pop();
    pPoint->show();

    MyStack<string> stringStack;
    string s = "C++";
    stringStack.push(s);
    stringStack.push("Zerohertz");
    cout << stringStack.pop() << '\t';
    cout << stringStack.pop() << endl;

    return 0;
}

template<class T>
MyStack<T>::MyStack(){
    tos = -1;
}
template<class T>
void MyStack<T>::push(T element){
    if(tos == 99){
        cout << "Stack full" << endl;
        return;
    }
    tos++;
    data[tos] = element;
}
template<class T>
T MyStack<T>::pop(){
    T Data;
    if(tos == -1){
        cout << "Stack empty" << endl;
        return 0;
    }
    Data = data[tos--];
    return Data;
}
~~~

>> Output

~~~C++
0	10	20	
(2,3)
(10,20)
Zerohertz	C++

Process finished with exit code 0
~~~

> gclass.cpp

~~~C++
#include <iostream>
using namespace std;

template<class T1, class T2>
class GClass{
    T1 data1;
    T2 data2;
public:
    GClass(){data1 = 0; data2 = 0;};
    void set(T1 a, T2 b){
        data1 = a; data2 = b;
    }
    void get(T1 &a, T2 &b){
        a = data1; b = data2;
    }
};

int main(){
    int a;
    double b;
    GClass<int, double> x;
    x.set(2, 0.5);
    x.get(a, b);
    cout << "a = " << a << "\tb = " << b << endl;

    char c;
    float d;
    GClass<char, float> y;
    y.set('m', 12.5);
    y.get(c, d);
    cout << "c = " << c << "\td = " << d << endl;

    return 0;
}
~~~

>> Output

~~~C++
a = 2	b = 0.5
c = m	d = 12.5

Process finished with exit code 0
~~~

## Quiz

> quiz.cpp

~~~C++
#include <iostream>
#include <string>
using namespace std;

class Shape{
protected:
    string name;
    int width, height;
public:
    Shape(string n = "", int w = 0, int h = 0){name = n; width = w; height = h;}
    virtual double getArea(){return 0;}
    string getName(){return name;}
};

class Oval : public Shape{
public:
    Oval(string n = "", int w = 0, int h = 0):Shape(n, w, h){}
    double getArea(){return 3.14 * width * height;}
};

class Rect : public Shape{
public:
    Rect(string n = "", int w = 0, int h = 0):Shape(n, w, h){}
    double getArea(){return width * height;}
};

class Triangular : public Shape{
public:
    Triangular(string n = "", int w = 0, int h = 0):Shape(n, w, h){}
    double getArea(){return width * height / 2;}
};

int main(){
    Shape *p[3];
    p[0] = new Oval("빈대떡", 10, 20);
    p[1] = new Rect("찰떡", 30, 40);
    p[2] = new Triangular("토스트", 30, 40);
    for(int i = 0; i < 3; i++)
        cout << p[i]->getName() << " 넓이는 " << p[i]->getArea() << endl;
    for(int i = 0; i < 3; i++) delete p[i];

    return 0;
}
~~~

>> Output

~~~C++
빈대떡 넓이는 628
찰떡 넓이는 1200
토스트 넓이는 600

Process finished with exit code 0
~~~

***

# Day 14

## STL

+ STL(Standard Template Library)
  + 표준 템플릿 라이브러리
  + 많은 제네릭 클래스와 제네릭 함수 포함
+ STL의 구성
  + 컨테이너 : 템플릿 클래스
    + 데이터를 담아두는 자료 구조를 표현한 클래스
    + 리스트, 큐, 스택, 맵, 셋, 벡터
  + iterator : 컨테이너 원소에 대한 포인터
    + 컨테이너의 원소들을 순회하면서 접근하기 위해 만들어진 컨테이너 원소에 대한 포인터
  + 알고리즘 : 템플릿 함수
    + 컨테이너 원소에 대한 복사, 검색, 삭제, 정렬 등의 기능을 구현한 템플릿 함수
    + 컨테이너의 멤버 함수 아님

> STL 컨테이너의 종류

|컨테이너 클래스|설명|헤더 파일|
|:-:|:-:|:-:|
|vector|동적 크기의 배열을 일반화한 클래스|`<vector>`|
|deque|앞뒤 모두 입력 가능한 큐 클래스|`<deque>`|
|list|빠른 삽입/삭제 가능한 리스트 클래스|`<list>`|
|set|정렬된 순서로 값을 저장하는 집합 클래스, 값은 유일|`<set>`|
|map|(key, value)쌍으로 값을 저장하는 맵 클래스|`<map>`|
|stack|스택을 일반화한 클래스|`<stack>`|
|queue|큐를 일반화한 클래스|`<queue>`|

> STL iterator의 종류

|iterator의 종류|iterator에 `++` 연산 후 방향|read/write|
|:-:|:-:|:-:|
|iterator|다음 원소로 전진|read/write|
|const_iterator|다음 원소로 전진|read|
|reverse_iterator|지난 원소로 후진|read/write|
|const_reverse_iterator|지난 원소로 후진|read|

> STL 알고리즘 함수들

+ copy
+ merge
+ random
+ rotate
+ equal
+ min
+ remove
+ search
+ find
+ move
+ replace
+ sort
+ max
+ partition
+ reverse
+ swap

## Vector

+ 가변 길이 배열을 구현한 `Generic` 클래스
+ 원소의 저장, 삭제, 검색 등 다양한 멤버 함수 지원
+ 벡터에 저장된 원소는 인덱스로 접근 가능

> vector.cpp

~~~C++
#include <iostream>
#include <vector>
using namespace std;

int main(){
    vector<int> v;

    v.push_back(1);
    v.push_back(2);
    v.push_back(3);

    for(int i = 0; i < v.size(); i++)
        cout << v[i] << '\t';
    cout << endl;

    v[0] = 10;
    int n = v[2];
    v.at(2) = 5;

    for(int i = 0; i < v.size(); i++)
        cout << v[i] << '\t';
    cout << endl;

    return 0;
}
~~~

>> Output

~~~C++
1	2	3	
10	2	5	

Process finished with exit code 0
~~~

## Iterator

+ 반복자라고도 부름
+ `*`, `++` 연산자 사용 가능
+ 컨테이너의 원소를 가리키는 포인터

> iterator.cpp

~~~C++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main(){
    vector<int> v;
    int n;

    cout << "5개의 정수를 입력하시오." << endl;
    for(int i = 0; i < 5; i++){
        cin >> n;
        v.push_back(n);
    }

    sort(v.begin(), v.end()); // sort(v.begin() + a, v.begin() + b) -> a에서 b - 1까지

    vector<int>::iterator it;

    for(it = v.begin(); it != v.end(); it++)
        cout << *it << '\t';
    cout << endl;

    return 0;
}
~~~

>> Output

~~~C++
5개의 정수를 입력하시오.
30 -7 250 6 120
-7	6	30	120	250	

Process finished with exit code 0
~~~

## Algorithm

+ 탐색(`find`) : 컨테이너 안에서 특정한 자료를 찾음
+ 정렬(`sort`) : 자료들을 크기 순으로 정렬
  + `param1` : 정렬을 시작한 원소의 주소
  + `param2` : 소팅 범위의 마지막 원소 다음 주소
+ 반전(`reverse`) : 자료들의 순서 역순
+ 삭제(`remove`) : 조건이 만족되는 자료 삭제
+ 변환(`transform`) : 컨테이너 요소들을 사용자가 제공하는 변환 함수에 따라 변환
 
> algorithm.cpp

~~~C++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Circle{
    string name;
    int radius;
public:
    Circle(int radius = 1, string name = ""):radius(radius), name(name){}
    double getArea(){return 3.14 * radius * radius;}
    string getName(){return name;}
    bool operator<(Circle b);
    friend ostream &operator<<(ostream &os, vector<Circle> &b);
};

void printVector(vector<Circle> vec);

int main(){
    vector<Circle> v;
    v.push_back(Circle(2, "waffle"));
    v.push_back(Circle(3, "pizza"));
    v.push_back(Circle(1, "donut"));
    v.push_back(Circle(5, "pizzaLarge"));
    printVector(v);
    // int it = v.size() - 1;
    sort(v.begin(), v.end()); // sort(&v[0], &v[it]);
    printVector(v);
    cout << endl << "프렌드함수 operator<<로 출력하는 경우" << endl;
    cout << v << endl;

    return 0;
}

bool Circle::operator<(Circle b){
    if(this->radius < b.radius)
        return true;
    else
        return false;
}

ostream &operator<<(ostream &os, vector<Circle> &b){
    vector<Circle>::iterator it;
    os << "모든 원소를 출력한다.>>";
    for(it = b.begin(); it != b.end(); it++)
        os << it->getName() << '\t';
    os << endl;
    return os;
}

void printVector(vector<Circle> vec){
    cout << "모든 원소를 출력한다.>>";
    for(auto it = vec.begin(); it != vec.end(); it++) // auto는 자동형변환
        cout << it->getName() << '\t';
    cout << endl;
}
~~~

>> Output

~~~C++
모든 원소를 출력한다.>>waffle   pizza   donut   pizzaLarge
모든 원소를 출력한다.>>donut    waffle  pizza   pizzaLarge

프렌드함수 operator<<로 출력하는 경우
모든 원소를 출력한다.>>donut    waffle  pizza   pizzaLarge
~~~