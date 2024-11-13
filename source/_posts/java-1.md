---
title: Java (1)
date: 2023-10-23 17:05:15
categories:
- 2. Backend
tags:
- Java
---
# Installation

```shell Install.sh
$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.6 LTS
Release:        20.04
Codename:       focal
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install openjdk-17-jdk
$ java -version
openjdk version "17.0.8.1" 2023-08-24
OpenJDK Runtime Environment (build 17.0.8.1+1-Ubuntu-0ubuntu120.04)
OpenJDK 64-Bit Server VM (build 17.0.8.1+1-Ubuntu-0ubuntu120.04, mixed mode, sharing)
$ javac -version
javac 17.0.8.1
$ sudo update-alternatives --config java
There is only one alternative in link group java (providing /usr/bin/java): /usr/lib/jvm/java-17-openjdk-amd64/bin/java
Nothing to configure.
$ vi ~/.zshrc
...
```

OS 버전 확인 및 업데이트 후 `sudo apt-get install openjdk-${VERSION}-jdk`로 JDK를 설치한다.
여기서 JDK는 Java Development Kit의 약자로 Java application을 개발하기 위한 software package다.
JDK는 개발, compile, debug 및 실행하는 데 필요한 모든 도구와 library를 포함한다.

<!-- More -->

1. Java Compiler (`javac`)
   + Java source code (`.java` 파일)를 Java bytecode (`.class` 파일)로 컴파일하는 도구
2. Java Virtual Machine (JVM)
   + Java bytecode를 실행하는 런타임 환경
   + `java` 명령어를 사용하여 JVM을 시작하고 Java application 실행 가능
3. Java Runtime Environment (JRE)
   + Java application을 실행하는 데 필요한 라이브러리와 종속성 포함
   + JVM과 함께 필수적인 class 라이브러리들 포함
   + 이를 통해 Java application 실행 가능
4. Java API
   + Java로 작성된 여러 기본 라이브러리와 class들로 구성
   + 개발자들이 일반적인 프로그래밍 작업을 수행하는 데 필요한 다양한 기능 제공
5. JDK Tools and Utilities
   + 개발, 디버깅, 모니터링 및 다른 작업을 위한 다양한 명령줄 도구 포함
   + Ex. `jdb` (Java 디버거), `jstat` (Java 통계 도구), `jhat` (힙 분석 도구) 등...

```bash ~/.zshrc
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export PATH=$JAVA_HOME/bin:$PATH
```

설치 후 사용하는 shell의 설정 파일에 위와 같이 환경 변수를 설정한다.
이는 `JAVA_HOME` 환경 변수를 설정하고 해당 경로를 시스템의 `PATH`에 추가하기 위함이다.
환경 변수를 추가하여 얻는 효과는 아래와 같다.

+ 여러 버전의 Java가 설치된 환경에서 특정 버전의 자바를 기본으로 사용하고자 할 때 `JAVA_HOME`을 적절히 설정해야 함
+ 많은 Java 기반 도구나 application은 `JAVA_HOME` 환경 변수 참조
+ `PATH`에 `JAVA_HOME/bin`을 추가함으로써, 해당 디렉터리의 실행 파일들을 전역적으로 접근 가능

```shell Install.sh
...
$ source ~/.zshrc
```

이렇게 변경한 환경 변수를 위의 명령어로 shell에 적용하면 끝이다.

```shell Uninstall.sh
$ sudo apt-get purge openjdk-\*
```

삭제 시 위의 명령어를 사용할 수 있다.

---

# Hello, World!

```java HelloWorld.java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

이 코드를 실행하려면 두 가지 방법이 존재한다.

1. `java HelloWorld.java`
   + Java 11 이후 버전에서 제공
   + 내부적으로 source code를 메모리에서 컴파일하고 바로 실행
   + `.class` 파일이 디스크에 생성되지 않음
2. `javac HelloWorld.java && java HelloWorld`
   + 전통적인 방식으로 source code를 컴파일하고 실행하는 방법
   + `javac HelloWorld.java`: `HelloWorld.class`라는 bytecode 파일 생성
   + `java HelloWorld`: bytecode를 JVM에서 실행
   + `.class` 파일이 디스크에 생성되므로, 필요한 경우 해당 bytecode를 다른 위치로 이동하거나 재사용 가능

이제 `HelloWorld.java`를 줄 마다 어떤 의미가 있는지 살펴보자.

```java
public class HelloWorld {
```

+ `public`
  + 접근 제한자
  + Class가 다른 package에서도 접근 가능함을 의미
+ `class`
  + Java에서 사용자 정의 데이터 타입을 생성하는 키워드

```java
    public static void main(String[] args) {
```

+ `static`
  + 정적 method임을 나타내는 키워드
  + 객체를 생성하지 않고 method 호출 가능
+ `void`
  + Method가 반환하는 자료형
+ `(String[] args)`
  + `main` method의 매개변수
  + Command line에서 전달되는 인자들을 배열로 입력

```java
        System.out.println("Hello, World!");
```

+ `System.out`
  + Java의 표준 출력 스트림
+ `println`
  + 표준 출력 스트림에 문자열 출력 후 줄 바꿈 수행 method

---

# Java Compilation Rules

1. Source File Structure
   - File Name
     - Java source 파일의 이름은 그 파일 내의 `public` class 이름과 동일해야 함
     - 만약 `public` class가 없다면 파일 내의 주 class와 일치하는 이름을 사용하는 것을 권장
   - One Public Class per File
     - 한 Java source 파일에는 단 하나의 `public` class만 있어야 함
2. Package Declaration
   - First Statement
     - package 선언은 source 파일의 첫 번째 문장이어야 함 (주석 및 공백 제외)
   - Directory Structure
     - Package 이름은 디렉토리 구조와 일치해야 함
     - Ex. `package com.example.test;`라는 선언이 있다면, source 파일은 `com/example/test` 디렉토리에 있어야 함
3. Import Statements
   - After Package
     - Import 문장은 package 선언 바로 다음에 나와야 함
   - No Duplicate Imports
     - 동일한 package나 class를 여러 번 임포트 할 필요는 없음
4. Class Declaration
   - One Public Class
     - 하나의 source 파일에는 단 하나의 `public` class만 있어야 함
   - Class Modifiers
     - class는 `public`, `abstract`, `final` 등의 수정자를 가질 수 있음
     - `private` 수정자는 최상위 class에는 사용할 수 없습니다.
5. Method and Field Declaration
   - Modifiers
     - method와 field는 다양한 접근 제어자와 수정자를 가질 수 있음
     - Ex. `public`, `private`, `protected`, `static`, `final` 등...
   - Order
     - Java 컴파일러는 method나 field의 선언 순서에 대해 특별한 요구사항을 가지고 있지 않음
     - 하지만 일반적인 코딩 관례에 따라 변수나 초기화 블록은 class의 상단에, method는 그 아래에 위치시키는 것을 권장
6. Other Compilation Units
   - Interfaces, Enums, and Annotations
     - 인터페이스, 열거형, 주석은 class와 유사한 규칙을 따름
     - Ex. `public` 인터페이스는 파일 이름과 동일한 이름을 가져야 함

예를 들어, 만약 `public class`의 이름과 파일 명이 다를 경우 아래와 같은 에러가 발생한다.

```shell
$ javac HelloWorld.java
HelloWorld.java:1: error: class HelloWorld2 is public, should be declared in a file named HelloWorld2.java
public class HelloWorld2 {
       ^
1 error
```

이 외에 Java의 여러 keyword들은 아래와 같다.

|키워드|설명|사용처 및 예시|
|:-:|:-:|:-:|
|(default)|아무 키워드도 없을 때. 같은 package 내에서만 접근 가능. (package-private)|class, method, field|
|`public`|어디서든 접근 가능|class, method, field|
|`protected`|같은 package나 상속받은 하위 class에서 접근 가능|method, field|
|`private`|같은 class 내에서만 접근 가능|method, field, 내부 class|
|`abstract`|구현되지 않은 class나 method. class는 인스턴스화될 수 없고, method는 하위 class에서 구현되어야 함|class, method|
|`final`|변경 불가능. class는 상속 불가, method는 오버라이드 불가, 변수/field는 한 번 할당 후 변경 불가|class, method, field/변수|
|`static`|인스턴스 없이 class 레벨에서 접근 가능. class 변수나 method는 모든 인스턴스가 공유|method, field, 내부 class|
|`synchronized`|여러 스레드가 동시에 접근할 때 해당 method나 블록의 코드 실행을 한 번에 하나의 스레드만 수행할 수 있게 동기화 함|method, 코드 블록|
|`transient`|객체 직렬화시, 해당 field의 값은 직렬화되지 않음|field|
|`volatile`|변수를 메인 메모리에서 읽고 쓰게 함. 여러 스레드가 동시에 사용할 때 변수의 일관성을 유지|field|
|`strictfp`|부동 소수점 연산의 결과가 플랫폼 간에 일관될 것을 보장|class, method|
|`native`|Java 외부의 C나 C++ 등 다른 언어로 작성된 method를 의미|method|

---

# Basic Grammar

## Const & Variable

상수 선언은 `final`을 자료형 앞에 사용하여 정의할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        final boolean a = true;
        System.out.println("a:\t" + a);
        final int b = 100;
        System.out.println("b:\t" + b);
        final double c = 3.14;
        System.out.println("c:\t" + c);
        final String d = "안녕하세요, 오효근입니다.";
        System.out.println("d:\t" + d);
    }
}
```

```shell
$ java Main.java
a:      true
b:      100
c:      3.14
d:      안녕하세요, 오효근입니다.
```

변수 선언과 초기화는 C와 매우 유사하게 구성되어 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        boolean a = true;
        System.out.println("a:\t" + a);
        int b = 100;
        System.out.println("b:\t" + b);
        double c = 3.14;
        System.out.println("c:\t" + c);
        String d = "안녕하세요, 오효근입니다.";
        System.out.println("d:\t" + d);
    }
}
```

```shell
$ java Main.java
a:      true
b:      100
c:      3.14
d:      안녕하세요, 오효근입니다.
```

## Operator

|Type|Operator|Mean|
|:-:|:-:|:-:|
|산술|`+`|더하기|
|산술|`-`|빼기|
|산술|`*`|곱하기|
|산술|`/`|나누기|
|산술|`%`|나머지 (모듈로)|
|산술|`++`|증가 (전위 또는 후위)|
|산술|`--`|감소 (전위 또는 후위)|
|비교|`==`|동일 (같음)|
|비교|`!=`|다름 (같지 않음)|
|비교|`>`|큼|
|비교|`<`|작음|
|비교|`>=`|크거나 같음|
|비교|`<=`|작거나 같음|
|논리|`&&`|논리 AND|
|논리|`\|\|`|논리 OR|
|논리|`!`|논리 NOT|
|비트|`&`|비트 AND|
|비트|`\|`|비트 OR|
|비트|`^`|비트 XOR (배타적 OR)|
|비트|`~`|비트 NOT|
|비트|`<<`|왼쪽 시프트|
|비트|`>>`|오른쪽 시프트 (부호 유지)|
|비트|`>>>`|오른쪽 시프트 (부호 무시, 0으로 채움)|
|대입|`=`|할당|
|대입|`+=`|더하고 할당|
|대입|`-=`|빼고 할당|
|대입|`*=`|곱하고 할당|
|대입|`/=`|나누고 할당|
|대입|`%=`|나머지 값으로 할당|
|대입|`&=`|비트 AND 후 할당|
|대입|`\|=`|비트 OR 후 할당|
|대입|`^=`|비트 XOR 후 할당|
|대입|`<<=`|왼쪽 시프트 후 할당|
|대입|`>>=`|오른쪽 시프트 후 할당|
|대입|`>>>=`|오른쪽 시프트 (부호 무시) 후 할당|
|특수|`?:`|삼항 연산자 (조건 연산자)|
|특수|`instanceof`|객체 타입 확인|
|특수|`()`, 타입 캐스팅|형 변환|

## Data Type & Type Conversion

+ Boolean: `boolean`
+ Numeric
  + Integer: `byte`, `short`, `int`, `long`
  + Float: `float`, `double`
+ String: `String`

Java에서 문자열은 `String` 클래스의 객체로 관리된다.
이는 문자열이 원시 자료형이 아니라 객체임을 의미한다.
문자열은 불변 (immutable)하므로, 기존 문자열을 변경하는 대신 새로운 문자열을 생성하게 된다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        String s = "test";
        System.out.println(s.getClass().getName());
    }
}
```

```shell
$ java Main.java
java.lang.String
```

예를 들어 문자열 내의 특정 문자를 변경하기 위해서는 아래와 같이 새로운 문자열을 만들어야한다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        String S = "asdfqwer";
        S = S.substring(0, 2) + "D" + S.substring(3);
        System.out.print(S);
    }
}
```

```shell
$ java Main.java
asDfqwer
```

또한 형변환은 아래와 같이 수행할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        int A = 10;
        float B = (float) A;
        String C = String.valueOf(A);
        System.out.println("A:\t" + ((Object) A).getClass().getName() + "\t" + A);
        System.out.println("B:\t" + ((Object) B).getClass().getName() + "\t\t" + B);
        System.out.println("C:\t" + C.getClass().getName() + "\t" + C);
    }
}
```

```shell
$ java Main.java
A:      java.lang.Integer       10
B:      java.lang.Float         10.0
C:      java.lang.String        10
```

## Conditional Statement

Java에서 조건문을 사용할 때 Go와 유사하지만 Go와는 다르게 `else if` 혹은 `else`가 중괄호 `} {` 사이에 존재할 필요는 없다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        int A = 10;
        if (A == 10) {
            System.out.println("Hi");
        } else {
            System.out.println("Bye");
        }
        A = 20;
        if (A == 10) {
            System.out.println("Hi");
        } else if (A == 12) {
            System.out.println("Hello");
        } else {
            System.out.println("Bye");
        }
    }
}
```

```shell
$ java Main.java
Hi
Bye
```

## Loop Statement

`for`와 `{` 사이의 반복에 대한 조건문에는 소괄호 (`( )`)가 필수로 존재해야한다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        for (int i = 0; i < 10; i++) {
            System.out.println(i + " Hi!");
        }
    }
}
```

```shell
$ java Main.java
0 Hi!
1 Hi!
2 Hi!
3 Hi!
4 Hi!
5 Hi!
6 Hi!
7 Hi!
8 Hi!
9 Hi!
```

---

# Formatter

VS Code에서 Java 파일을 저장할 때 [이렇게](https://code.visualstudio.com/docs/java/java-linting) formatting 하려면 `settings.json` 파일에 아래 코드들을 추가하면 된다.

```json settings.json
{
    ...
    "editor.formatOnSave": true,
    "java.format.settings.url": "https://raw.githubusercontent.com/google/styleguide/gh-pages/eclipse-java-google-style.xml",
    "java.format.settings.profile": "GoogleStyle",
    ...
}
```