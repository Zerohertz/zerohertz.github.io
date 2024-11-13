---
title: Java (3)
date: 2024-08-28 00:25:56
categories:
  - 2. Backend
tags:
  - Java
---

# Package

> Package: 비슷한 성격의 class들을 모아 놓은 Java의 directory

Java package를 구성하기 위하여 아래와 같은 directory 구조를 생성했다.

```shell
$ tree
.
├── animals
│   ├── Animal.java
│   ├── cat
│   │   ├── Cat.java
│   │   ├── KoreanShotHair.java
│   │   └── RussianBlue.java
│   └── dog
│       ├── BullDog.java
│       ├── Dog.java
│       └── Husky.java
├── Main.java
└── zerohertz
    └── Zerohertz.java
```

위 예제에선 `animals`와 `zerohertz`라는 2개의 package로 구성되어 있고, `animals` package 내에 `cat`과 `dog`라는 2개의 subpackage가 존재한다.
각 package 내부의 code들은 아래와 같다.

<!-- More -->

```java
// animals/Animal.java
package animals;

public class Animal {
}

// animals/cat/Cat.java
package animals.cat;

import animals.Animal;

public class Cat extends Animal {
}

// animals/cat/KoreanShotHair.java
package animals.cat;

public class KoreanShotHair extends Cat {
}

// animals/cat/RussianBlue.java
package animals.cat;

public class RussianBlue extends Cat {
}

// animals/dog/Dog.java
package animals.dog;

import animals.Animal;

public class Dog extends Animal {
}

// animals/dog/BullDog.java
package animals.dog;

public class BullDog extends Dog {
}

// animals/dog/Dog.java
package animals.dog;

public class Husky extends Dog {
}

// zerohertz/Zerohertz.java
package zerohertz;

import animals.cat.Cat;
import animals.dog.Dog;

public class Zerohertz {
    public static void hello(Cat cat) {
        System.out.println("😻");
    }

    public static void hello(Dog dog) {
        System.out.println("🐶");
    }
}
```

Java package를 구성하기 위해 `package` keyword를 사용한다.
이렇게 구성한 package는 아래와 같이 `import` keyword를 사용하여 불러올 수 있다.

```java Main.java
import animals.cat.KoreanShotHair;
import animals.dog.BullDog;
import animals.dog.Dog;
import zerohertz.Zerohertz;

public class Main {
    public static void main(String[] args) {
        KoreanShotHair koreanShotHair = new KoreanShotHair();
        Dog bullDog = new BullDog();
        Zerohertz.hello(koreanShotHair);
        Zerohertz.hello(bullDog);
    }
}
```

```shell
$ javac *.java && java Main.java
😻
🐶
```

---

# Access Modifier

> [Access modifier](https://docs.oracle.com/javase/tutorial/java/javaOO/accesscontrol.html): Access level modifiers determine whether other classes can use a particular field or invoke a particular method.

| Access Modifier | Same Class | Same Package | Subclass (different package) | Any Class |
| --------------- | ---------- | ------------ | ---------------------------- | --------- |
| `private`       | Yes        | No           | No                           | No        |
| `default`       | Yes        | Yes          | No                           | No        |
| `protected`     | Yes        | Yes          | Yes                          | No        |
| `public`        | Yes        | Yes          | Yes                          | Yes       |

---

# Static

> [Static](https://docs.oracle.com/javase/tutorial/java/javaOO/classvars.html): The use of the static keyword to create fields and methods that belong to the class, rather than to an instance of the class.

Static에 대해 알아보기 위하여 아래 에제를 살펴보자.
숫자를 세기위한 `Counter` class에 대해 `counter1`, `counter2`라는 2개의 instance를 생성한다.

```java Counter.java
public class Counter {
    int count = 0;

    public void add() {
        this.count += 1;
    }

    public int getCount() {
        return this.count;
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Counter counter1 = new Counter();
        Counter counter2 = new Counter();
        print(counter1, counter2);
        counter1.add();
        print(counter1, counter2);
        counter2.add();
        print(counter1, counter2);
    }

    private static void print(Counter counter1, Counter counter2) {
        System.out.printf("%d\t%d\n", counter1.getCount(), counter2.getCount());
    }
}
```

```shell
$ javac *.java && java Main.java
0       0
1       0
1       1
```

위 결과를 통해 `Counter`의 property `count`는 공유되지 않고 instance 별로 존재하는 것을 확인할 수 있다.

## Property

그렇다면 `count`에 `static` keyword를 추가하면 어떻게 될까?

```java Counter.java
public class Counter {
    static int count = 0;

    public void add() {
        count += 1;
    }

    public int getCount() {
        return count;
    }
}
```

```shell
$ javac *.java && java Main.java
0       0
1       1
2       2
```

위와 같이 2개의 instance임에도 불구하고 각각 동일한 `count`를 공유함을 확인할 수 있다.

## Method

여기서 `getCount` method 또한 `static` keyword로 정의하면 아래와 같이 사용할 수 있고 instance 생성 없이 method를 호출할 수 있다.

```java Counter.java
public class Counter {
    static int count = 0;

    public void add() {
        count += 1;
    }

    public static int getCount() {
        return count;
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Counter counter1 = new Counter();
        Counter counter2 = new Counter();
        print();
        counter1.add();
        print();
        counter2.add();
        print();
        System.out.println(counter1 == counter2);
    }

    private static void print() {
        System.out.println(Counter.getCount());
    }
}
```

```shell
$ javac *.java && java Main.java
0
1
2
false
```

## Singleton Pattern

`static` keyword를 통해 singleton pattern을 구현한다면 아래와 같이 사용할 수 있다.

```java Counter.java
public class Counter {
    static int count = 0;
    private static Counter single;

    private Counter() {
    }

    public static Counter getInstance() {
        if (single == null) {
            single = new Counter();
        }
        return single;
    }

    public void add() {
        count += 1;
    }

    public static int getCount() {
        return count;
    }
}
```

```java Main.java
public class Main {
    public static void main(String[] args) {
        Counter counter1 = Counter.getInstance();
        Counter counter2 = Counter.getInstance();
        print();
        counter1.add();
        print();
        counter2.add();
        print();
        System.out.println(counter1 == counter2);
    }

    private static void print() {
        System.out.println(Counter.getCount());
    }
}
```

```shell
$ javac *.java && java Main.java
0
1
2
true
```

Constructor method `Counter`를 `private`으로 선언하고 instance 생성을 위한 `getInstance` method를 `public`으로 선언한다.
결과적으로 항상 하나의 instance만을 가지게 되므로 `counter1`과 `counter2`의 memory 주소가 동일한 것을 확인할 수 있다.

---

# Exception

기본적으로 Java에서 예외를 처리하기 위해 아래와 같이 `try-catch`를 사용할 수 있다.

```java ZeroException.java
public class ZeroException extends RuntimeException {
}
```

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

        int input = Integer.valueOf(br.readLine());
        try {
            test(input);
            System.out.println("[TRY]\t\tDone!");
        } catch (ZeroException error) {
            System.err.printf("[CATCH]\t\tException: %s\n", error);
        } finally {
            System.out.println("[FINALLY]\tDone!");
        }
    }

    private static void test(int input) throws ZeroException {
        if (input == 0) {
            throw new ZeroException();
        }
    }
}
```

```shell
$ javac *.java && java Main.java
0
[CATCH]         Exception: ZeroException
[FINALLY]       Done!
$ javac *.java && java Main.java
1
[TRY]           Done!
[FINALLY]       Done!
```

위의 출력 결과에서 알 수 있듯 `try`에서 예외가 발생하면 `catch`에 속한 code들이 실행되며 `finally`에 속한 code들은 예외 발생 여부에 상관 없이 실행된다.
또한 `throw`를 통하여 예외를 발생시킬 수 있고 `throws`를 통해 method 선언부에서 처리하지 않은 예외를 호출자에게 전달할 수 있다.

---

# Thread

Java에서 thread는 `Thread` class를 상속하여 아래와 같이 사용할 수 있다.

```java Main.java
public class Main extends Thread {
    int sequence;

    public Main(int sequence) {
        this.sequence = sequence;
    }

    public static void main(String[] args) {
        System.out.println("main Start");
        for (int i = 0; i < 5; i++) {
            Main main = new Main(i);
            main.start();
        }
        System.out.println("main End");
    }

    public void run() {
        System.out.printf("Thread %d Start\n", this.sequence);
        try {
            Thread.sleep(1000);
        } catch (Exception error) {
        }
        System.out.printf("Thread %d End\n", this.sequence);
    }
}
```

```shell
$ java Main.java
main Start
Thread 0 Start
Thread 4 Start
main End
Thread 3 Start
Thread 2 Start
Thread 1 Start
Thread 0 End
Thread 4 End
Thread 3 End
Thread 2 End
Thread 1 End
```

출력을 확인해보면 thread가 모두 실행되기 전에 `main` method가 먼저 종료된 것을 확인할 수 있다.
이러한 현상을 방지하기 위하여 생성된 thread를 `ArrayList`에 저장하고 `join` method를 호출하여 thread가 종료될 때까지 기다리게 할 수 있다.

```java Main.java
import java.util.ArrayList;

public class Main extends Thread {
    int sequence;

    public Main(int sequence) {
        this.sequence = sequence;
    }

    public static void main(String[] args) {
        System.out.println("main Start");
        ArrayList<Thread> threads = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Main main = new Main(i);
            main.start();
            threads.add(main);
        }
        for (int i = 0; i < threads.size(); i++) {
            Thread thread = threads.get(i);
            try {
                thread.join();
            } catch (Exception error) {
            }
        }
        System.out.println("main End");
    }

    public void run() {
        System.out.printf("Thread %d Start\n", this.sequence);
        try {
            Thread.sleep(1000);
        } catch (Exception error) {
        }
        System.out.printf("Thread %d End\n", this.sequence);
    }
}
```

```shell
$ java Main.java
main Start
Thread 0 Start
Thread 4 Start
Thread 3 Start
Thread 2 Start
Thread 1 Start
Thread 0 End
Thread 4 End
Thread 2 End
Thread 3 End
Thread 1 End
main End
```

하지만 위 code의 `Main` class는 `Thread` class에 대해 상속받았기 때문에 차후 변경 시 문제가 될 수 있다.
따라서 `Runnable` interface를 아래와 같이 사용할 수 있다.

```java Main.java
import java.util.ArrayList;

public class Main implements Runnable {
    int sequence;

    public Main(int sequence) {
        this.sequence = sequence;
    }

    public static void main(String[] args) {
        System.out.println("main Start");
        ArrayList<Thread> threads = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Thread main = new Thread(new Main(i));
            main.start();
            threads.add(main);
        }
        for (int i = 0; i < threads.size(); i++) {
            Thread thread = threads.get(i);
            try {
                thread.join();
            } catch (Exception error) {
            }
        }
        System.out.println("main End");
    }

    public void run() {
        System.out.printf("Thread %d Start\n", this.sequence);
        try {
            Thread.sleep(1000);
        } catch (Exception error) {
        }
        System.out.printf("Thread %d End\n", this.sequence);
    }
}
```

`Thread` class의 상속을 받지 않고 interface를 사용하기 때문에 `Main` class의 `start` method는 존재하지 않는다.
따라서 `Thread main = new Thread(new Main(i))`와 같이 instance를 생성하여 `start` method를 사용한다.

---

# Lambda

Java에서 lambda를 통해 함수형 programming style로 아래와 같이 개발할 수 있다.

```java Calculator.java
@FunctionalInterface
public interface Calculator {
    int add(int a, int b);
}
```

```java Calculator.java
public class Main {
    public static void main(String[] args) {
        Calculator calculator = (a, b) -> a + b;
        System.out.println(calculator.add(1, 1));
    }
}
```

```shell
$ javac *.java && java Main.java
2
```

Lambda 표현식은 단 하나의 abstract method를 가진 함수형 interface에서만 사용할 수 있으며, `@FunctionalInterface` annotation을 사용하여 명시적으로 함수형 interface임을 선언할 수 있다.

---

# Reference

- [Jump to Java](https://wikidocs.net/book/31)
