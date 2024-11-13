---
title: Java (2)
date: 2024-08-22 00:04:14
categories:
  - 2. Backend
tags:
  - Java
---

# Data Types

## String

String이란 문자들이 순서대로 나열된 일련의 문자 sequence를 의미한다.
Java에선 아래와 같이 `String`을 선언할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        String str1 = "Zerohertz";
        System.out.println(str1);
        String str2 = new String("Zerohertz");
        System.out.println(str2);
    }
}
```

```shell
$ java Main.java
Zerohertz
Zerohertz
```

<!-- More -->

하지만 여기서 `str1`과 `str2`는 다른 방식으로 선언된 것을 확인할 수 있는데, 아래와 같은 차이점이 존재한다.

|           Feature           |                        `str1`                         |                   `str2`                   |
| :-------------------------: | :---------------------------------------------------: | :----------------------------------------: |
| Memory Allocation Location  |                      String Pool                      |                Heap Memory                 |
|       Object Creation       | 동일한 literal이 존재하면 새로운 객체를 생성하지 않음 |           항상 새로운 객체 생성            |
| Reference Comparison (`==`) |           동일한 literal이 있을 경우 `true`           |                항상 `false`                |
|      Memory Efficiency      |                         More                          |                    Less                    |
|        Usage Method         |                  String literal 사용                  |      `new` 키워드를 사용한 객체 생성       |
|      Typical Use Case       |        동일한 문자열을 여러 번 사용할 때 유리         | 특별한 이유로 새로운 객체가 필요할 때 사용 |

> [Literal](https://www.baeldung.com/java-literals#what-is-a-java-literal): Any value we specify as a constant in the code.

Primitive (원시) 자료형 (`int`, `float`, `boolean`, `char`, ...)은 `new` keyword를 통해 선언할 수 없고 literal 표기 방식을 통해 선언할 수 있다.
하지만 primitive 자료형을 제외하고 `String`은 유일하게 `new` keyword를 사용하지 않고 literal 표기 방식을 통해 선언할 수 있다.
또한 `String` 내에서 아래와 같은 method들을 사용할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        String str1 = "Zerohertz";
        String str2 = new String("Zerohertz");
        System.out.println(str1 == str2);                                // false
        System.out.println(str1.equals(str2));                           // true
        System.out.println(str1.indexOf("hertz"));                       // 4
        System.out.println(str1.contains("hertz"));                      // true
        System.out.println(str1.contains("java"));                       // false
        System.out.println(str1.charAt(3));                              // o
        System.out.println(str1.replaceAll("e", "x"));                   // Zxrohxrtz
        System.out.println(str1.substring(0, 4));                        // Zero
        System.out.println(str1.toUpperCase());                          // ZEROHERTZ
        System.out.println(str1.toLowerCase());                          // zerohertz
        print(str1.split("e"));                                          // Z       roh     rtz
        System.out.println(String.format("String: %s", "Zerohertz"));    // String: Zerohertz
        System.out.println(String.format("Integer: %d", 1023));          // Integer: 1023
        System.out.println(String.format("Float: %f", 10.23));           // Float: 10.230000
        System.out.println(String.format("Float: %.2f", 10.23));         // Float: 10.23
        System.out.printf("Float: %.2f", 10.23);                         // Float: 10.23
    }

    private static void print(Object[] array) {
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i] + "\t");
        }
        System.out.println();
    }
}
```

하지만 `String`은 immutable인 한계점이 있기 때문에 `StringBuffer`와 `StringBuilder`를 사용한다.
이에 대한 차이점은 아래와 같다.

|              Feature              |       `String`        |                 `StringBuffer`                 |              `StringBuilder`              |
| :-------------------------------: | :-------------------: | :--------------------------------------------: | :---------------------------------------: |
|            Mutability             |       Immutable       |                    Mutable                     |                  Mutable                  |
| Synchronization<br />(Thread-safe) |          ❌           |                       ⭕️                       |                    ❌                     |
|             Use Case              | 변경되지 않는 문자열  | Multithread 환경에서 문자열 변경이 필요한 경우 | 단일 thread에서 문자열 변경이 필요한 경우 |
|            Performance            | 변경이 적을 때 효율적 |      동기화로 인해 성능이 상대적으로 낮음      |     동기화가 없기 때문에 성능이 높음      |
|         초기화 후 변경 시         |  새로운 객체를 생성   |         동일 객체 내에서 문자열을 수정         |      동일 객체 내에서 문자열을 수정       |
|           Java Version            |     모든 version      |                  JDK 1.0 이상                  |               JDK 1.5 이상                |

아래와 같이 `String`에서 추가적인 문자열을 추가하기 위해서는 `+=`을 사용할 수 있지만 매번 `String` 객체를 생성해야하기 때문에 효율적이지 않다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        String string = "";
        string += "Zero";
        string += "hertz";
        System.out.printf("String:\t\t%s\n", string);

        StringBuffer stringBuffer = new StringBuffer();
        stringBuffer.append("Zero");
        stringBuffer.append("hertz");
        System.out.printf("StringBuffer:\t%s\n", stringBuffer);

        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Zero");
        stringBuilder.append("hertz");
        System.out.printf("StringBuilder:\t%s\n", stringBuilder);
    }
}
```

```shell
$ java Main.java
String:         Zerohertz
StringBuffer:   Zerohertz
StringBuilder:  Zerohertz
```

`StringBuffer`와 `StringBuilder`는 `insert` method 로 특정 index에 문자열을 삽입하고 `substring` method로 문자열을 slicing 할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Zero");
        stringBuilder.append("hertz");
        stringBuilder.insert(4, "INSERT");
        System.out.println(stringBuilder);
        System.out.println(stringBuilder.substring(4, 10));
    }
}
```

```shell
$ java Main.java
ZeroINSERThertz
INSERT
```

## Array

> Array (배열): 고정된 크기로 요소들을 순차적으로 저장하는 data structure

```java Main.java
public class Main {
    public static void main(String[] args) {
        Object[] arrayObj = { 1, 1.0, "1" };
        print(arrayObj);
    }

    private static void print(Object[] array) {
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i] + "\t");
        }
        System.out.println();
    }
}
```

```shell
$ java Main.java
1       1.0     1
```

Index를 통해 값을 조회하는 것은 `[]`을, array의 길이는 `length` property를 사용할 수 있다.

## List

> List: 동적인 크기로 요소들을 순차적으로 저장하는 data structure

Java에서 list는 아래와 같이 `java.util.ArrayList`를 통해 사용할 수 있다.

```java Main.java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class Main {
    public static void main(String[] args) {
        ArrayList<Object> listObj = new ArrayList<>();
        listObj.add(1);
        listObj.add("1");
        listObj.add(1, 2);
        print(listObj);                                   // 1       2       1
        System.out.println(listObj.contains("1"));        // true
        System.out.println(listObj.contains("2"));        // false
        System.out.println(listObj.remove("1"));          // true
        print(listObj);                                   // 1       2

        // ArrayList<int> list = new ArrayList<>();
        // int is primitive type
        ArrayList<Integer> listInt = new ArrayList<>();
        listInt.add(1);
        listInt.add(4);
        listInt.add(2);
        print(listInt);                                   // 1       4       2
        listInt.sort(Comparator.naturalOrder());
        print(listInt);                                   // 1       2       4
        listInt.sort(Comparator.reverseOrder());
        print(listInt);                                   // 4       2       1

        ArrayList<String> listStr = new ArrayList<>(Arrays.asList("1", "2", "3"));
        System.out.println(String.join(",", listStr));    // 1,2,3
    }

    private static <T> void print(ArrayList<T> list) {
        for (int i = 0; i < list.size(); i++) {
            System.out.print(list.get(i) + "\t");
        }
        System.out.println();
    }
}
```

`add` method에 하나의 parameter를 입력하면 우측에 추가되고, 두개의 parameter를 입력하면 특정 index에 값을 삽입할 수 있다.
Array와 다르게 index를 통해 값을 조회하는 것은 `get` method를, list의 길이는 `size` method를 이용해야한다.
위 code에서 `<>`는 list 내 요소들의 type을 지정하는 기능인 [generic type](https://docs.oracle.com/javase/tutorial/java/generics/types.html)이다.
만약 `listObj`와 같이 `Object`로 선언 후 특정 index의 요소를 변수로 지정하면 아래와 같은 오류가 발생한다.

```shell
$ java Main.java
Main.java:14: error: incompatible types: Object cannot be converted to int
        int test = listObj.get(1);
                              ^
```

따라서 `int test = (int) listObj.get(1);`와 같이 명시적인 형변환을 수행하거나 list 자체를 `listInt`와 같이 선언 시 `<>`를 사용해야 이러한 오류가 발생하지 않는다.

## Map

> Map: Key와 value의 쌍을 저장하는 data structure

Java에서 map은 아래와 같이 `java.util.HashMap`을 통해 사용할 수 있다.

```java Main.java
import java.util.HashMap;

public class Main {
    public static void main(String[] args) {
        HashMap<Object, Integer> map = new HashMap<>();
        map.put("1", 1);
        map.put(2, 2);
        System.out.println(map);                         // {1=1, 2=2}
        System.out.println(map.size());                  // 2
        System.out.println(map.keySet());                // [1, 2]
        System.out.println(map.get("1"));                // 1
        System.out.println(map.get("2"));                // null
        System.out.println(map.getOrDefault("2", 2));    // 2
        System.out.println(map.containsKey("1"));        // true
        System.out.println(map.containsKey("2"));        // false
        System.out.println(map.remove("1"));             // 1
        System.out.println(map);                         // {2=2}
    }
}
```

`put` method와 `get` method를 통해 map 내에 key, value 쌍을 입력할 수 있다.
존재하지 않는 key에 대해 `get` method를 사용하면 `null`이 출력되지만 `getOrDefault` method를 사용하면 기본 값을 지정할 수 있다.

## Set

> Set: 중복되지 않는 요소들의 집합을 저장하는 data structure

```java Main.java
import java.util.Arrays;
import java.util.HashSet;

public class Main {
    public static void main(String[] args) {
        HashSet<Object> set1 = new HashSet<>();
        set1.add(1);
        set1.add("1");
        set1.add(1);
        System.out.println(set1);                            // [1, 1]
        HashSet<Object> set2 = new HashSet<>(Arrays.asList(1, 2, 3));

        HashSet<Object> intersection = new HashSet<>(set1);
        System.out.println(intersection.retainAll(set2));    // true
        System.out.println(intersection);                    // [1]

        HashSet<Object> union = new HashSet<>(set1);
        System.out.println(union.addAll(set2));              // true
        System.out.println(union);                           // [1, 1, 2, 3]

        HashSet<Object> substract = new HashSet<>(set1);
        System.out.println(substract.removeAll(set2));       // true
        System.out.println(substract);                       // [1]
    }
}
```

`add` method를 통하여 `HashSet`에 값을 추가할 수 있고, `remove` method를 통하여 `HashSet`에 존재하는 값을 삭제할 수 있다.

## Enum

> Enum: 열거형 type을 정의하며, 이 type에 속하는 상수 값들을 지정

```java Main.java
public class Main {
    enum Alphabet {
        a, b, c
    }

    private static void test(Alphabet arg) {
        System.out.println(arg);
    }

    public static void main(String[] args) {
        test(Alphabet.a);
    }
}
```

```shell
$ java Main.java
a
```

## Type Conversion

Type conversion은 아래와 같은 method들로 실행할 수 있다.

| Method                         | Description                                                                | Example                                     |
| ------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------- |
| `Integer.parseInt(String)`     | Converts a `String` to `int`                                               | `int num = Integer.parseInt("123");`        |
| `Integer.valueOf(String)`      | Converts a `String` to an `Integer` object                                 | `Integer num = Integer.valueOf("123");`     |
| `Double.parseDouble(String)`   | Converts a `String` to `double`                                            | `double d = Double.parseDouble("123.45");`  |
| `Double.valueOf(String)`       | Converts a `String` to a `Double` object                                   | `Double d = Double.valueOf("123.45");`      |
| `Float.parseFloat(String)`     | Converts a `String` to `float`                                             | `float f = Float.parseFloat("12.34f");`     |
| `Float.valueOf(String)`        | Converts a `String` to a `Float` object                                    | `Float f = Float.valueOf("12.34f");`        |
| `Long.parseLong(String)`       | Converts a `String` to `long`                                              | `long l = Long.parseLong("123456789");`     |
| `Long.valueOf(String)`         | Converts a `String` to a `Long` object                                     | `Long l = Long.valueOf("123456789");`       |
| `Boolean.parseBoolean(String)` | Converts a `String` to `boolean`                                           | `boolean b = Boolean.parseBoolean("true");` |
| `Boolean.valueOf(String)`      | Converts a `String` to a `Boolean` object                                  | `Boolean b = Boolean.valueOf("true");`      |
| `String.valueOf(int)`          | Converts a primitive `int` to a `String`                                   | `String s = String.valueOf(123);`           |
| `String.valueOf(double)`       | Converts a primitive `double` to a `String`                                | `String s = String.valueOf(123.45);`        |
| `String.valueOf(float)`        | Converts a primitive `float` to a `String`                                 | `String s = String.valueOf(12.34f);`        |
| `String.valueOf(long)`         | Converts a primitive `long` to a `String`                                  | `String s = String.valueOf(123456789L);`    |
| `String.valueOf(boolean)`      | Converts a primitive `boolean` to a `String`                               | `String s = String.valueOf(true);`          |
| `toString()`                   | Converts an object to a `String`<br />(generally available for all objects) | `String s = Integer.toString(123);`         |
| `(int) variable`               | Casts another primitive type to `int`                                      | `int i = (int) 123.45;`                     |
| `(double) variable`            | Casts another primitive type to `double`                                   | `double d = (double) 123;`                  |
| `(float) variable`             | Casts another primitive type to `float`                                    | `float f = (float) 123.45;`                 |
| `(long) variable`              | Casts another primitive type to `long`                                     | `long l = (long) 123.45;`                   |
| `Object.toString()`            | Converts an object to a `String`<br />(can be overridden for all objects)   | `String s = obj.toString();`                |

---

# Class

기본적으로 Java에서 class를 정의하기 위해서 아래와 같이 구성할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        Animal cat = new Animal();
        System.out.println(cat.name);
        cat.name = "dust";
        System.out.println(cat.name);
    }
}

class Animal {
    String name;
}
```

```shell
$ java Main.java
null
dust
```

하지만 Java에서 [`public class`는 각 `.java` file 내에 하나만 존재](https://docs.oracle.com/javase/tutorial/java/package/createpkgs.html)해야하기 때문에 아래와 같이 구성한다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        Animal cat = new Animal();
        System.out.println(cat.getName());
        cat.setName("dust");
        System.out.println(cat.getName());
    }
}
```

```java Animal.java
public class Animal {
    private String name;

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
```

```shell
$ javac *.java && java Main.java
null
dust
```

여기서 첫 code와 다르게 `name`이 `private`으로 설정되어있고, getter와 setter인 `getName` method와 `setName` method가 추가된 것을 확인할 수 있다.
이는 [object-oriented programming (OOP)의 원칙인 encapsulation을 지키기 위하여 필수적](https://docs.oracle.com/javaee/6/tutorial/doc/gjbbp.html)이기 때문이다.

## Inheritance

Java의 inheritance (상속)는 `extends` keyword를 아래와 같이 사용한다.

```java Cat.java
public class Cat extends Animal {
    public void bark() {
        System.out.println("Meow");
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Cat cat = new Cat();
        cat.setName("dust");
        System.out.println(cat.getName());
        cat.bark();
    }
}
```

```shell
$ javac *.java && java Main.java
dust
Meow
```

또한 Java에서는 [multiple inheritance (다중 상속)](https://docs.oracle.com/javase/tutorial/java/IandI/multipleinheritance.html)을 허용하지 않는다.

## Overriding

부모 class의 method를 자식 class가 동일한 입출력의 형태로 method를 구현하는 것을 [method overriding](https://docs.oracle.com/javase/tutorial/java/IandI/override.html)이라 하며 아래와 같이 구현할 수 있다.

```java RussianBlue.java
public class RussianBlue extends Cat {
    @Override
    public void bark() {
        System.out.println("Meow!!!");
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        RussianBlue cat = new RussianBlue();
        cat.setName("dust");
        System.out.println(cat.getName());
        cat.bark();
    }
}
```

```shell
$ javac *.java && java Main.java
dust
Meow!!!
```

여기서 `@Override`는 [annotation](https://docs.oracle.com/javase/tutorial/java/annotations/index.html)의 한 종류이며 compiler에 정보를 전달하여 예기치 못한 오류를 방지할 수 있기에 사용이 권장된다.

## Overloading

한 class의 둘 이상 method의 이름이 동일하며 입출력이 다른 경우 [method overloading](https://docs.oracle.com/javase/specs/jls/se14/html/jls-8.html#jls-8.4.9)이라 하며 아래와 같이 구현할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        System.out.println(add(1, 2));
        System.out.println(add("Zero", "hertz"));
    }

    public static int add(int a, int b) {
        return a + b;
    }

    public static String add(String c, String d) {
        return c + d;
    }
}
```

```shell
$ java Main.java
3
Zerohertz
```

## Constructor

Java의 constructor (생성자)는 아래와 같이 class의 이름을 method와 같이 구현하면 된다.
Constructor를 구현할 때 `void`를 포함한 `return` type은 정의하지 않으며 overloading 또한 가능하다.
만약 constructor가 class 내에 존재하지 않다면 [compiler는 비어있는 default constructor를 생성](https://docs.oracle.com/javase/tutorial/java/javaOO/constructors.html)한다.

```java Cat.java
public class Cat extends Animal {
    public Cat(String name) {
        this.setName(name);
    }

    public void bark() {
        System.out.println("Meow");
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Cat cat = new Cat("dust");
        System.out.println(cat.getName());
    }
}
```

```shell
$ javac *.java && java Main.java
dust
```

## Interface

> [Interface](https://docs.oracle.com/javase/specs/jls/se7/html/jls-9.html): An interface declaration introduces a new reference type whose members are classes, interfaces, constants, and abstract methods. This type has no implementation, but otherwise unrelated classes can implement it by providing implementations for its abstract methods.

Interface를 왜 사용하는지 알아보기 위해 아래 예제를 살펴보자.

```java Cat.java
public class Cat extends Animal {
}
```

```java Dog.java
public class Dog extends Animal {
}
```

```java Zerohertz.java
public class Zerohertz {
    public static void call(Cat cat) {
        System.out.println("Hello, Cat!");
    }

    public static void call(Dog dog) {
        System.out.println("Hello, Dog!");
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Cat cat = new Cat();
        Dog dog = new Dog();
        Zerohertz.call(cat);
        Zerohertz.call(dog);
    }
}
```

```shell
$ javac *.java && java Main.java
Hello, Cat!
Hello, Dog!
```

위 code는 간단히 구현할 수 있지만, 동물 class가 100개가 된다면 `Zerohertz` class의 `static` method를 100개 구현해야할 것이다.
하지만 interface를 아래와 같이 구현하여 사용한다면 하나의 `static` method로 100개의 class에 대해 기능을 구현할 수 있다.

```java Cute.java
public interface Cute {
}
```

```java Cat.java
public class Cat extends Animal implements Cute {
}
```

```java Dog.java
public class Dog extends Animal implements Cute {
}
```

```java Zerohertz.java
public class Zerohertz {
    public static void call(Cute cute) {
        System.out.printf("Hello, %s!\n", cute.getClass().getSimpleName());
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Cat cat = new Cat();
        Dog dog = new Dog();
        Zerohertz.call(cat);
        Zerohertz.call(dog);
    }
}
```

```shell
$ javac *.java && java Main.java
Hello, Cat!
Hello, Dog!
```

Interface는 상속을 받은 class임에도 불구하고 부모 class에 의존적이지 않은 독립적 class가 될 수 있는 장점이 존재한다.
이러한 역할 때문에 interface는 constructor를 가질 수 없으며 상수 (`public static final`), abstract method (`public abstract`), default method (`default`), static method (`static`), private method (`private`) 만을 정의할 수 있다.
이에 대해 더 자세히 알아보기 위해 아래 예제를 살펴보자.

```java Cute.java
public interface Cute {
    // public static final
    int CUTE_SCORE = 100;

    // abstract method
    void eat();

    // default method
    default void grooming() {
        System.out.println("Can't Grooming!!!");
    }

    // static method
    static void hello() {
        System.out.println("I'm so cute!");
    }

    // private method
    private int getCuteScore() {
        if (this instanceof Cat) {
            return CUTE_SCORE * 2;
        }
        return CUTE_SCORE;
    }

    // default method
    default void printCuteScore() {
        System.out.printf("Cute score of %s: %s\n", this.getClass().getSimpleName(), getCuteScore());
    }
}
```

귀여운 동물들의 interface를 구현하기 위해 얼마나 귀여운지 정량적으로 평가하기 위해 `CUTE_SCORE`를, 먹는 것을 구현하기 위해 abstract method `eat`을, 인사를 위한 static method `hello`를, `CUTE_SCORE`의 getter인 private method `getCuteScore`를, `CUTE_SCORE`를 출력하기 위한 default method `printCuteScore`를 개발했다.
~~(고양이가 좀 귀엽기 때문에 `getCuteScore` method에서 `CUTE_SCORE` 2배)~~

```java Cat.java
public class Cat extends Animal implements Cute {
    @Override
    public void eat() {
        System.out.println("Fish");
    }

    @Override
    public void grooming() {
        System.out.println("😻");
    }
}
```

고양이는 생선을 먹으므로 `"Fish"`를 출력하고, grooming을 하기 때문에 `😻`를 출력한다.

```java Dog.java
public class Dog extends Animal implements Cute {
    @Override
    public void eat() {
        System.out.println("Meat");
    }
}
```

개는 고기를 먹으므로 `"Meat"`를 출력하고, grooming을 하지 않기 때문에 따로 구현하지 않았다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        Cat cat = new Cat();
        Dog dog = new Dog();
        cat.eat();               // Fish
        dog.eat();               // Meat
        cat.grooming();          // 😻
        dog.grooming();          // Can't Grooming!!!
        Cute.hello();            // I'm so cute!
        cat.printCuteScore();    // Cute score of Cat: 200
        dog.printCuteScore();    // Cute score of Dog: 100
    }
}
```

결과적으로 위와 같은 출력을 확인할 수 있다.

## Polymorphism

> [Polymorphism](https://docs.oracle.com/javase/tutorial/java/IandI/polymorphism.html): Subclasses of a class can define their own unique behaviors and yet share some of the same functionality of the parent class.

Polymorphism (다형성)에 대해 더 자세히 알아보기 위해 아래 예제를 살펴보자.

```java
// Dog.java
public class Dog extends Animal {
}

// Cute.java
public interface Cute {
}

// OutDoor.java
public interface OutDoor {
    int getSpeed();
}

// CuteOutDoor.java
public interface CuteOutDoor extends Cute, OutDoor {
}

// Husky.java
public class BullDog extends Dog implements OutDoor {
    public int getSpeed() {
        return 50;
    }
}

// Beagle.java
public class Beagle extends Dog implements Cute, OutDoor {
    public int getSpeed() {
        return 100;
    }
}

// Husky.java
public class Husky extends Dog implements CuteOutDoor {
    public int getSpeed() {
        return 200;
    }
}

// Zerohertz.java
public class Zerohertz {
    public static void call(Cute cute) {
        System.out.printf("Hello, %s!\n", cute.getClass().getSimpleName());
    }

    public static void walk(OutDoor outdoor) {
        System.out.printf("Let's walk, %s! (Speed: %d)\n", outdoor.getClass().getSimpleName(), outdoor.getSpeed());
    }
}
```

불독은 산책을 할 수 있지만 귀엽지 않다. ~~(실제로는 귀여움)~~
반면 비글과 허스키는 산책을 할 수 있고 귀엽다.
이러한 특성들을 위와 같이 interface와 inheritance를 이용하여 구현할 수 있다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        BullDog bullDog = new BullDog();
        // Zerohertz.call(bullDog);
        // Main.java:4: error: incompatible types: BullDog cannot be converted to Cute
        //         Zerohertz.call(bullDog);
        //                        ^
        // Note: Some messages have been simplified; recompile with -Xdiags:verbose to get full output
        // 1 error
        Zerohertz.walk(bullDog);

        Beagle beagle = new Beagle();
        Zerohertz.call(beagle);
        Zerohertz.walk(beagle);

        Husky husky = new Husky();
        Zerohertz.call(husky);
        Zerohertz.walk(husky);
    }
}
```

불독은 귀엽지 않기 때문에 `Zerohertz.call` method를 사용하면 오류가 발생한다.

```shell
$ javac *.java && java Main.java
Let's walk, BullDog! (Speed: 50)
Hello, Beagle!
Let's walk, Beagle! (Speed: 100)
Hello, Husky!
Let's walk, Husky! (Speed: 200)
```

이 예제에서 `husky`는 `Husky` class의 객체이며 `Dog` class 및 `Animal` class의 객체이고 `Cute` interface와 `OutDoor` interface의 객체이다.
이렇게 하나의 객체가 여러 data type을 가지는 것을 polymorphism이라 하며, 이러한 특성으로 `Zerohertz.walk` method에서 `husky`가 입력되면 data type을 `OutDoor`로 바꾸어 사용할 수 있다.

---

# I/O

## Console Input

Console 입력을 받기 위하여 기본적으로 `java.io.InputStream` class를 사용할 수 있다.

```java Main.java
import java.io.IOException;
import java.io.InputStream;

public class Main {
    public static void main(String[] args) throws IOException {
        InputStream in = System.in;

        int value;
        value = in.read();
        System.out.println(value);
    }
}
```

```shell
$ java Main.java
asdf
97
```

하지만 `InputStream`의 `read` method는 1byte 크기의 입력만을 받고 `int` (ASCII)로 저장되기 때문에 한계점이 존재한다.
이러한 한계점을 극복하기 위하여 `java.io.InputStreamReader`를 아래와 같이 사용할 수 있다.

```java Main.java
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) throws IOException {
        InputStream in = System.in;
        InputStreamReader reader = new InputStreamReader(in);

        char[] value = new char[3];
        reader.read(value);
        System.out.println(value);
    }
}
```

```shell
$ java Main.java
asdf
asd
```

하지만 `InputStreamReader`는 고정된 크기를 지정해야하기 때문에 한계점이 존재한다.
이러한 한계점을 극복하기 위해 `String`으로 입력을 받을 수 있는 `java.io.BufferedReader`를 아래와 같이 사용할 수 있다.

```java Main.java
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) throws IOException {
        InputStream in = System.in;
        InputStreamReader reader = new InputStreamReader(in);
        BufferedReader br = new BufferedReader(reader);

        String value = br.readLine();
        System.out.println(value);
    }
}
```

```shell
$ java Main.java
asdf
asdf
```

하지만 `BufferedReader`를 사용하기 위해서 4개의 class를 `import` 해야하는 한계점이 존재한다.
이러한 한계점을 극복하기 위해 `java.util.Scanner`를 아래와 같이 사용한다.

```java Main.java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String value = sc.next();
        System.out.println(value);
        sc.close();
    }
}
```

```shell
$ java Main.java
asdf
asdf
```

그럼에도 불구하고 `Scanner`는 `BufferedReader`에 비하여 느린 단점이 존재한다.
따라서 `BufferedReader`와 `Scanner`는 상황에 맞게 사용해야한다.

## Console Output

`System.err`은 `System.out`과 유사한 역할을 수행하지만 오류 message를 출력할 때 사용하는 차이점이 존재한다.

```java Main.java
public class Main {
    public static void main(String[] args) {
        System.out.println("System.out.println");
        System.err.println("System.err.println");
    }
}
```

```shell
$ java Main.java > out.log 2> err.log
$ cat out.log
System.out.println
$ cat err.log
System.err.println
```

## File Writing

File을 Java로 작성하기 위해 아래와 같이 `java.io.FileWriter` 또는 `java.io.PrintWriter`를 사용할 수 있다.

```java Main.java
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class Main {
    public static void main(String[] args) throws IOException {
        FileWriter fw = new FileWriter("tmp.log");
        for (int i = 0; i < 5; i++) {
            String data = i + "-Line\n";
            fw.write(data);
        }
        fw.close();

        PrintWriter pw = new PrintWriter(new FileWriter("tmp.log", true));
        for (int i = 0; i < 5; i++) {
            String data = i + "-Line";
            pw.println(data);
        }
        pw.close();
    }
}
```

```shell
$ java Main.java
$ cat tmp.log
0-Line
1-Line
2-Line
3-Line
4-Line
0-Line
1-Line
2-Line
3-Line
4-Line
```

## File Reading

File을 Java로 읽기 위해 아래와 같이 `java.io.FileReader`와 `java.io.BufferedReader`를 사용할 수 있다.

```java Main.java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("tmp.log"));
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            System.out.println(line);
        }
        br.close();
    }
}
```

```shell
$ java Main.java
0-Line
1-Line
2-Line
3-Line
4-Line
0-Line
1-Line
2-Line
3-Line
4-Line
```

---

# Reference

- [Jump to Java](https://wikidocs.net/book/31)
