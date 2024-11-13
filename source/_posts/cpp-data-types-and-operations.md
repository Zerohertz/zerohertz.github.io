---
title: 'C++: Data Types and Operations'
date: 2020-07-01 16:24:22
categories:
- Etc.
tags:
- C, C++
---
# 변수와 상수

+ 변수(Variable) : 프로그램이 실행되는 동안 저장된 값이 변경될 수 있는 공간
+ 상수(Constant) : 프로그램이 실행되는 동안 값이 변경되지 않는 공간

***

# 자료형

> 데이터의 종류를 자료형(Data type) 또는 데이터 타입이라고 한다.

+ 변수는 일단 하나의 자료형으로 정의되면 해당되는 종류의 데이터만 저장할 수 있음
+ 자료형을 크게 나누면 정수형(Interger type)과 문자형(Charater type), 부동 소수점형(Floating-point type)으로 나눌 수 있음
  + 정수형 : 정수 타입의 데이터
  + 문자형 : 하나의 문자 데이터
  + 부동 소수점형 : 실수 타입의 데이터

<!-- More -->

***

# 변수 선언

+ 변수를 사용하기 전에 반드시 미리 선언
+ 변수 선언 : 컴파일러에게 어떤 변수를 사용하겠다고 미리 알리는 것

> 변수 선언 방법

~~~C++
자료형 변수이름
char c;
int i;
double interest_rate;
~~~

+ 변수가 선언되면 변수의 값은 아직 정의되지 않은 상태
+ 변수를 선언과 동시에 값을 넣는 방법은 변수 이름 뒤에 할당 연산자 `=`를 놓고 초기값을 적어 넣으면 됨 - 변수의 초기화(Initialization)

> 변수의 초기화

~~~C++
char c = 'a';
int i = 7;
double interest_rate = 0.05;
~~~

***

# 정수형

+ `short` : 16비트(2바이트)
+ `int` : 32비트(4바이트)
+ `long` : 32비트(4바이트)
+ `sizeof()` : 자료형의 크기를 Return하는 함수
+ `unsigned` : 자료형이 음수가 아닌 값만을 나타낸다는 것을 의미(더 넓은 범위의 양수)
+ `signed` : 자료형이 음수도 가질 수 있음을 명백히 하는데 사용(`signed int == int`, 보통 생략)

## 오버플로우

> Overflow.cpp

~~~C++
#include <iostream>
#include <climits>
using namespace std;

int main(){
    short s = SHRT_MAX;
    unsigned short u = USHRT_MAX;
    
    cout << "원래 s 값 : " << s << endl;
    cout << "원래 u 값 : " << u << endl;
    
    s = s + 1;
    u = u + 1;
    
    cout << "오버플로우 s 값 : " << s << endl;
    cout << "오버플로우 u 값 : " << u << endl;
    return 0;
}
~~~

> Output

~~~C++
원래 s 값 : 32767
원래 u 값 : 65535
오버플로우 s 값 : -32768
오버플로우 u 값 : 0
Program ended with exit code: 0
~~~

+ 변수가 나타낼 수 있는 범위가 제한되어 있기 때문에 일어남
+ 오버플로우가 발생하더라도 컴파일러는 아무런 경고를 하지 않음

## 기호 상수

~~~C++
# define TAX_RATE 0.15
const double TAX_RATE = 0.15;
~~~

+ `#define`
  + 보통 프로그램의 맨 첫 부분에 모여 있음
  + 컴파일러가 동작하기 전에 전처리기(Preprocessor)가 처리
  + 전처리기는 단순히 기호의 이름을 전부 찾아 정의된 값으로 바꿈(`TAX_RATE`를 전부 찾아 `0.15`로 바꿈)
+ `const`
  + `const`를 변수 선언 앞에 붙이면 상수가 됨
  + 선언 시 `const`가 붙여진 변수는 일단 초기화 된 후 그 값이 변경될 수 없음
  + 변수 선언과 같이 끝에 `;` 사용

***

# 문자형

+ 문자(Character)는 한글이나 영어에서의 하나의 글자, 숫자, 기호 등을 의미
+ 문자를 숫자로 나타내는 규격으로 아스키(ASCII : American Standard Code for Information Interchange) 존재

> Char.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    char c;
    cin >> c;
    cout << "원래의 C : " << c << endl;
    
    c = c + 1;
    cout << "C + 1 : " << c << endl;
    
    return 0;
}
~~~

> Output

~~~C++
a
원래의 C : a
C + 1 : b
Program ended with exit code: 0

...

원래의 C : A
C + 1 : B
Program ended with exit code: 0
~~~

## Bool형

+ 참(`true`) or 거짓(`false`)

***

# 부동 소수점형

|자료형|명칭|크기|범위|
|:-:|:-:|:-:|:-:|
|float|단일정밀도(Single-precision) 부동 소수점|32비트|$\pm1.17549\times10^{-38}$~$\pm3.40282\times10^{+38}$|
|double|두배정밀도(Double-precision) 부동 소수점|64비트|$\pm2.22507\times10^{-308}$~$\pm1.79769\times10^{+308}$|
|long double|두배확장정밀도(Double-extension-precision) 부동 소수점|64비트 또는 80비트|$\pm2.22507\times10^{-308}$~$\pm1.79769\times10^{+308}$|

> Float.cpp

~~~C++
#include <iostream>
#include <string>
using namespace std;

int main(){
    cout.setf(ios_base::fixed); //소수점 표기법
    
    float fvalue = 1234567890.12345678901234567890;
    double dvalue = 1234567890.12345678901234567890;
    
    cout << "Float : " << fvalue << endl;
    cout << "Double : " << dvalue << endl;
    
    return 0;
}
~~~

> Output

~~~C++
Float : 1234567936.000000
Double : 1234567890.123457
Program ended with exit code: 0
~~~

> 소수점 표기법을 사용하지 않은 Output(`cout.setf()` 삭제)

~~~C++
Float : 1.23457e+09
Double : 1.23457e+09
Program ended with exit code: 0
~~~

> Overflow.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    float x = 1e39;
    cout << x <<endl;
    return 0;
}
~~~

> Output

~~~C++
inf
Program ended with exit code: 0
~~~

> Underflow.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    float x = 1.23e-39;
    float y = 1.23e-46;
    cout << "X : " << x << endl;
    cout << "y : " << y << endl;
    return 0;
}
~~~

> Output

~~~C++
X : 1.23e-39
y : 0
Program ended with exit code: 0
~~~

***

# 수식과 연산자

+ 수식 : 피연산자들과 연산자의 조합
+ 연산자(Operator) : 어떤 연산을 나타내는 기호
+ 피연산자(Operand) : 연산의 대상이 되는 것

|연산자의 분류|연산자|의미|
|:-:|:-:|:-:|
|할당|`=`|오른쪽을 왼쪽에 할당|
|산술|`+` `-` `*` `/` `%`|사칙연산과 나머지 연산|
|부호|`+` `-`|양수와 음수 표시|
|증감|`++` `--`|증가, 감소 연산|
|관계|`>` `<` `==` `!=` `>=` `<=`|오른쪽과 왼쪽 비교|
|논리|`&&` `||` `!`|논리적인 AND, OR|
|조건|`?`|조건에 따라 선택|
|콤마|`,`|피연산자들을 순차적으로 실행|
|비트 단위 연산자|`&` `|` `^` `~` `<<` `>>`|비트별 AND, OR, XOR, 이동, 반전|
|sizeof 연산자|`sizeof`|자료형이나 변수의 크기를 바이트 단위로 반환|
|형변환|`(type)`|변수나 상수의 자료형을 변환|
|포인터 연산자|`*` `&` `[]`|주소계산, 포인터가 가르키는 곳의 내용 추출|
|구조체 연산자|`.` `->`|구조체의 멤버 참조|

***

# 산술 연산자

+ `%` : 나머지 값
+ `x += y` : `x = x + y`
+ `++x` : `x`값을 먼저 증가한 후에 다른 연산에 사용
+ `x++` : `x`값을 먼저 사용한 후에 증가

## 형변환

+ 자동적인 형변환
  + 할당 연산자의 오른쪽에 있는 값은 왼쪽에 있는 변수의 자료형으로 자동 변환
+ 명시적인 형변환
  + `(자료형) 상수 또는 변수`

> Type.cpp

~~~C++
#include <iostream>
using namespace std;

int main(){
    int a = 10;
    float b = 10.82;
    cout << "int + float : " << a + b << endl;
    cout << "float + int : " << b + a << endl;
    cout << (int)a + (int)b << endl;
    return 0;
}
~~~

> Output

~~~C++
int + float : 20.82
float + int : 20.82
20
Program ended with exit code: 0
~~~

***

# 관계 연산자

+ 관계 연산자(Relational operator) : 두 개의 피연산자를 비교하는데 사용
+ `==`, `!=`, `>`, `<`, `>=`, `<=`

***

# 논리 연산자

+ 논리 연산자 : 여러 개의 조건을 조합하여 참인지 거짓인지를 따짐
+ `&&`, `||`, `!`

***

# 조건 연산자

+ `exp1 ? exp2 : exp3`
  + `exp1`이 참이면 결과값은 `exp2`
  + `exp1`이 거짓이면 결과값은 `exp3`