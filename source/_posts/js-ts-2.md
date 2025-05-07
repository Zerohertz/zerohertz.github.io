---
title: JavaScript & TypeScript (2)
date: 2025-05-07 20:58:55
categories:
  - Etc
tags:
  - JavaScript
  - TypeScript
---

# Conditional Statement

TypeScript는 JavaScript와 동일한 조건문 구문을 사용하지만, type checking을 통해 code의 안전성을 높인다.

## if-else Statement

기본적인 조건 분기 구문으로, 조건에 따라 다른 code block을 실행한다.

```ts
// 기본 if-else 구문
const age = 25;

if (age >= 18) {
  console.log("Adult");
} else {
  console.log("Minor");
}

// 여러 조건 확인 (if-else if-else)
const score = 85;

if (score >= 90) {
  console.log("Grade: A");
} else if (score >= 80) {
  console.log("Grade: B");
} else if (score >= 70) {
  console.log("Grade: C");
} else {
  console.log("Grade: F");
}

// TypeScript에서의 type narrowing
function process(value: string | number) {
  if (typeof value === "string") {
    // 이 block 내에서 value는 string type으로 처리됨
    return value.toUpperCase();
  } else {
    // 이 block 내에서 value는 number type으로 처리됨
    return value.toFixed(2);
  }
}
```

<!-- More -->

## Ternary Operator

간단한 조건식을 한 줄로 표현할 수 있는 연산자이다.

```ts
// 기본 구문: condition ? expressionIfTrue : expressionIfFalse
const age = 20;
const status = age >= 18 ? "Adult" : "Minor";

// 중첩 ternary (가독성을 위해 줄바꿈)
const score = 85;
const grade = score >= 90 ? "A" : score >= 80 ? "B" : score >= 70 ? "C" : "F";

// TypeScript에서 type에 따른 분기
type User = { name: string; role: "admin" | "user" };
const user: User = { name: "Kim", role: "admin" };
const message = user.role === "admin" ? "Welcome, Admin" : "Welcome, User";
```

## Switch Statement

여러 case를 검사해야 할 때 if-else 보다 가독성이 좋을 수 있다.

```ts
// 기본 switch 구문
const day = new Date().getDay();
let dayName: string;

switch (day) {
  case 0:
    dayName = "Sunday";
    break;
  case 1:
    dayName = "Monday";
    break;
  case 2:
    dayName = "Tuesday";
    break;
  case 3:
    dayName = "Wednesday";
    break;
  case 4:
    dayName = "Thursday";
    break;
  case 5:
    dayName = "Friday";
    break;
  case 6:
    dayName = "Saturday";
    break;
  default:
    dayName = "Invalid day";
    break;
}

// TypeScript literal union type과 함께 사용
type Direction = "north" | "east" | "south" | "west";
const direction: Direction = "east";

switch (direction) {
  case "north":
    console.log("Going up");
    break;
  case "east":
    console.log("Going right");
    break;
  case "south":
    console.log("Going down");
    break;
  case "west":
    console.log("Going left");
    break;
  // 모든 case를 처리했으므로 default가 필요 없음
  // TypeScript는 이를 검증할 수 있음
}
```

## Nullish Coalescing

`null` 또는 `undefined` 값을 다룰 때 유용한 `??` 연산자를 제공한다.

```ts
// nullish coalescing 연산자 (??)
const input = null;
const value = input ?? "default value"; // "default value"

// optional chaining과 함께 사용
type User = {
  name: string;
  settings?: {
    theme?: string;
  };
};

const user: User = { name: "Kim" };
const theme = user.settings?.theme ?? "light"; // "light"
```

## Type Guards

TypeScript에서는 type 검사를 위한 다양한 guard가 있다.

```ts
// typeof type guard
function process(value: string | number) {
  if (typeof value === "string") {
    return value.toUpperCase();
  }
  return value.toFixed(2);
}

// instanceof type guard
class Animal {
  move() {
    console.log("Moving");
  }
}
class Bird extends Animal {
  fly() {
    console.log("Flying");
  }
}

function handleAnimal(animal: Animal) {
  animal.move();
  if (animal instanceof Bird) {
    animal.fly(); // Bird instance인 경우에만 접근 가능
  }
}

// custom type guard (type predicates)
interface Fish {
  swim: () => void;
}
interface Bird {
  fly: () => void;
}

// 'pet is Fish'는 type predicate
function isFish(pet: Fish | Bird): pet is Fish {
  return (pet as Fish).swim !== undefined;
}

function move(pet: Fish | Bird) {
  if (isFish(pet)) {
    pet.swim(); // pet은 Fish type으로 좁혀짐
  } else {
    pet.fly(); // pet은 Bird type으로 좁혀짐
  }
}
```

## Discriminated Unions

Tagged union이라고도 하며, TypeScript의 강력한 pattern 중 하나이다.

```ts
// discriminated union 예시
interface Circle {
  kind: "circle"; // discriminant
  radius: number;
}

interface Square {
  kind: "square"; // discriminant
  sideLength: number;
}

type Shape = Circle | Square;

function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      // 이 block에서 shape는 Circle type
      return Math.PI * shape.radius ** 2;
    case "square":
      // 이 block에서 shape는 Square type
      return shape.sideLength ** 2;
  }
}
```

{% cq %}
TypeScript에서 조건문은 JavaScript와 동일하게 작동하지만, type system의 이점을 활용하여 더 안전하고 명확한 code를 작성할 수 있다.
특히 type guard와 discriminated union pattern을 적절히 활용하면 runtime error를 크게 줄일 수 있다.
{% endcq %}

---

# Loop Statement

TypeScript는 JavaScript의 모든 반복문을 지원하며, type checking을 통해 더 안전한 iteration을 구현할 수 있다.

## `for` Loop

가장 기본적인 반복문으로, 초기화, 조건, 증감식으로 구성된다.

```ts
// 기본 for loop
for (let i = 0; i < 5; i++) {
  console.log(i); // 0, 1, 2, 3, 4
}

// Array와 함께 사용
const numbers: number[] = [10, 20, 30, 40, 50];
for (let i = 0; i < numbers.length; i++) {
  console.log(numbers[i]); // 10, 20, 30, 40, 50
}

// TypeScript에서는 index와 value의 type이 자동으로 추론됨
const names: string[] = ["Kim", "Lee", "Park"];
for (let i = 0; i < names.length; i++) {
  // i는 number, names[i]는 string으로 추론됨
  const name: string = names[i]; // type annotation 필요 없음
  console.log(`${i}: ${name}`);
}
```

## `for...of` Loop

ES6에서 도입된 이 loop는 iterable object의 각 요소를 순회한다.

```ts
// Array 순회 (value에 접근)
const fruits: string[] = ["Apple", "Banana", "Cherry"];
for (const fruit of fruits) {
  console.log(fruit); // "Apple", "Banana", "Cherry"
  // fruit은 자동으로 string으로 추론됨
}

// String 순회 (character 단위)
const message: string = "Hello";
for (const char of message) {
  console.log(char); // "H", "e", "l", "l", "o"
  // char는 string으로 추론됨
}

// Map 순회
const userAge = new Map<string, number>([
  ["Kim", 30],
  ["Lee", 25],
  ["Park", 35],
]);

for (const [name, age] of userAge) {
  // destructuring과 함께 사용
  console.log(`${name}: ${age}`); // "Kim: 30", "Lee: 25", "Park: 35"
  // name은 string, age는 number로 추론됨
}

// TypeScript에서 custom iterable 사용
class NumberRange implements Iterable<number> {
  constructor(
    private start: number,
    private end: number,
  ) {}

  *[Symbol.iterator](): Iterator<number> {
    for (let i = this.start; i <= this.end; i++) {
      yield i;
    }
  }
}

const range = new NumberRange(1, 5);
for (const num of range) {
  console.log(num); // 1, 2, 3, 4, 5
  // num은 number로 추론됨
}
```

## `for...in` Loop

object의 enumerable property를 순회한다.
`Array`보다는 `object`에 사용하는 것이 권장된다.

```ts
// object의 key 순회
const user = {
  name: "Kim",
  age: 30,
  role: "admin",
};

for (const key in user) {
  console.log(`${key}: ${user[key as keyof typeof user]}`);
  // TypeScript에서는 key를 적절한 type으로 변환해야 함
}

// Array에 사용 시 주의사항
const numbers = [10, 20, 30];
numbers.customProp = "test"; // Array에 custom property 추가

for (const key in numbers) {
  console.log(key); // "0", "1", "2", "customProp"
  // index 뿐만 아니라 property도 순회됨
}
```

## `while` Loop

조건이 참인 동안 code block을 반복 실행한다.

```ts
// 기본 while loop
let count = 0;
while (count < 5) {
  console.log(count); // 0, 1, 2, 3, 4
  count++;
}

// Array 처리
const items: string[] = ["a", "b", "c"];
let index = 0;
while (index < items.length) {
  console.log(items[index]); // "a", "b", "c"
  index++;
}

// TypeScript type guard와 함께 사용
function processValues(values: (string | number)[]) {
  let i = 0;
  while (i < values.length) {
    const value = values[i];
    if (typeof value === "string") {
      // value는 이 block에서 string으로 처리
      console.log(value.toUpperCase());
    } else {
      // value는 이 block에서 number로 처리
      console.log(value.toFixed(2));
    }
    i++;
  }
}
```

## `do...while` Loop

Code block을 최소 한 번 실행한 후, 조건이 참인 동안 반복한다.

```ts
// 기본 do...while loop
let count = 0;
do {
  console.log(count); // 0, 1, 2, 3, 4
  count++;
} while (count < 5);

// 조건이 처음부터 거짓인 경우에도 한 번은 실행됨
let num = 10;
do {
  console.log("이 코드는 한 번 실행됩니다.");
  num++;
} while (num < 10);

// 사용자 입력 처리 예시 (가상 코드)
function getValidInput(): number {
  let input: number;
  do {
    const rawInput = prompt("양수를 입력하세요") || "";
    input = Number(rawInput);
  } while (isNaN(input) || input <= 0);

  return input;
}
```

## `Array` 고차 함수

TypeScript에서는 Array의 고차 함수를 사용할 때 callback의 parameter type이 자동으로 추론된다.

```ts
// forEach - array의 각 요소에 대해 함수 실행
const numbers: number[] = [1, 2, 3, 4, 5];
numbers.forEach((num) => {
  // num은 자동으로 number로 추론됨
  console.log(num * 2); // 2, 4, 6, 8, 10
});

// map - 새로운 array 반환
const doubled = numbers.map((num) => num * 2);
// doubled는 number[] type으로 추론됨
console.log(doubled); // [2, 4, 6, 8, 10]

// filter - 조건에 맞는 요소만 추출
const evens = numbers.filter((num) => num % 2 === 0);
// evens는 number[] type으로 추론됨
console.log(evens); // [2, 4]

// reduce - array를 단일 값으로 축소
const sum = numbers.reduce((acc, num) => acc + num, 0);
// sum은 number type으로 추론됨
console.log(sum); // 15

// TypeScript의 type 안전성
interface User {
  id: number;
  name: string;
  active: boolean;
}

const users: User[] = [
  { id: 1, name: "Kim", active: true },
  { id: 2, name: "Lee", active: false },
  { id: 3, name: "Park", active: true },
];

// 활성 사용자의 이름만 추출
const activeUserNames = users
  .filter((user) => user.active)
  .map((user) => user.name);
// activeUserNames는 string[] type으로 추론됨
console.log(activeUserNames); // ["Kim", "Park"]
```

## `break` & `continue`

반복문의 흐름을 제어하는 keyword이다.

```ts
// break - 반복문 즉시 종료
for (let i = 0; i < 10; i++) {
  if (i === 5) break;
  console.log(i); // 0, 1, 2, 3, 4
}

// continue - 현재 iteration 건너뛰기
for (let i = 0; i < 10; i++) {
  if (i % 2 === 0) continue;
  console.log(i); // 1, 3, 5, 7, 9
}

// nested loop에서 label 사용
outerLoop: for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    if (i === 1 && j === 1) {
      break outerLoop; // 외부 loop까지 종료
    }
    console.log(`(${i}, ${j})`);
    // (0, 0), (0, 1), (0, 2), (1, 0)만 출력됨
  }
}
```

## Iterator와 Generator

TypeScript는 ES6의 iterator와 generator를 완벽하게 지원한다.

```ts
// Iterator 사용
function createRangeIterator(start: number, end: number): Iterator<number> {
  let current = start;
  return {
    next() {
      return current <= end
        ? { value: current++, done: false }
        : { value: undefined, done: true };
    },
  };
}

const rangeIter = createRangeIterator(1, 3);
let result = rangeIter.next();
while (!result.done) {
  console.log(result.value); // 1, 2, 3
  result = rangeIter.next();
}

// Generator 함수 사용
function* rangeGenerator(start: number, end: number): Generator<number> {
  for (let i = start; i <= end; i++) {
    yield i;
  }
}

const gen = rangeGenerator(1, 3);
for (const num of gen) {
  console.log(num); // 1, 2, 3
  // TypeScript는 num을 자동으로 number로 추론
}

// async generator 예시
async function* fetchPages(): AsyncGenerator<string, void, unknown> {
  for (let i = 1; i <= 3; i++) {
    const response = await fetch(`https://api.example.com/page${i}`);
    const text = await response.text();
    yield text;
  }
}

// 비동기 iteration (가상 코드)
async function processPages() {
  for await (const pageContent of fetchPages()) {
    console.log(pageContent.length);
  }
}
```

{% cq %}
TypeScript에서 반복문은 JavaScript의 모든 기능을 제공하면서도 type 안전성을 추가한다.
특히 collection 처리에서는 `for...of`와 Array 고차 함수를 활용하면 가독성이 높고 type-safe한 code를 작성할 수 있다.
Generator와 iterator를 통해 memory 효율적인 data 처리도 가능하다.
{% endcq %}

---

# Function

```ts
// Function Declaration
function add1(a: number, b: number): number {
  return a + b;
}

// Function Expression
const add2 = function (a: number, b: number): number {
  return a + b;
};

// Arrow Function Expression
const add3 = (a: number, b: number): number => {
  return a + b;
};
```

| 특징                | 함수 선언식<br/>(Function Declaration)                                                                                                                                                                                                                | 함수 표현식<br/>(Function Expression)                                                                                                                                                                                                        | 화살표 함수<br/>(Arrow Function)                                                                                                                                                                                                                                                                                                                                                                                                         |
| :------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hoisting            | O<br/>(전체 함수가 hoisting됨)                                                                                                                                                                                                                        | X<br/>(변수 선언부만 hoisting, 할당은 X)                                                                                                                                                                                                     | X<br/>(변수 선언부만 hoisting, 할당은 X)                                                                                                                                                                                                                                                                                                                                                                                                 |
| `this` binding      | 호출 방식에 따라 동적으로 결정됨<br/>(Dynamic `this`)                                                                                                                                                                                                 | 호출 방식에 따라 동적으로 결정됨<br/>(Dynamic `this`)                                                                                                                                                                                        | 선언될 때의 상위 scope의 `this`를 가짐<br/>(Lexical `this`)                                                                                                                                                                                                                                                                                                                                                                              |
| `arguments` 객체    | O<br/>(함수 내에서 사용 가능)                                                                                                                                                                                                                         | O<br/>(함수 내에서 사용 가능)                                                                                                                                                                                                                | X<br/>(상위 스코프의 `arguments`를 참조하거나, rest 파라미터 `...args` 사용)                                                                                                                                                                                                                                                                                                                                                             |
| `new` 키워드 사용   | O<br/>(생성자 함수로 사용 가능)                                                                                                                                                                                                                       | O<br/>(생성자 함수로 사용 가능, 익명 함수는 주로 X)                                                                                                                                                                                          | X<br/>(생성자 함수로 사용 불가, `prototype` 속성이 없음)                                                                                                                                                                                                                                                                                                                                                                                 |
| Method로 사용 시    | `this`는 호출한 객체를 가리킴                                                                                                                                                                                                                         | `this`는 호출한 객체를 가리킴                                                                                                                                                                                                                | `this`는 상위 scope를 가리키므로, 객체의 method로 사용할 때 주의 필요<br/>(객체 literal 내에서는 객체를 가리키지 않음)                                                                                                                                                                                                                                                                                                                   |
| 가독성 / 간결성     | 일반적인 함수 형태                                                                                                                                                                                                                                    | 함수를 변수에 할당하는 형태로, callback 등에 유용                                                                                                                                                                                            | 구문이 간결하며, 특히 callback 함수에 유용                                                                                                                                                                                                                                                                                                                                                                                               |
| 익명 함수 가능 여부 | X<br/>(반드시 이름이 있어야 함)                                                                                                                                                                                                                       | O<br/>(익명 함수 가능, 변수에 할당)                                                                                                                                                                                                          | O<br/>(주로 익명 형태로 사용, 변수에 할당)                                                                                                                                                                                                                                                                                                                                                                                               |
| 주요 사용 상황      | - Code의 주요 구성 block으로 명확하게 함수를 정의할 때<br/>- 전역적으로 사용되거나, 다른 여러 곳에서 호출될 함수<br/>- Hoisting의 이점을 활용하고 싶을 때<br/>(함수 선언 위치에 구애받지 않고 호출)<br/>- 생성자 함수로 사용할 때<br/>(전통적인 방식) | - 함수를 값으로 다뤄야 할 때<br/>(변수에 할당, 객체의 속성으로 정의)<br/>- Callback 함수로 전달할 때<br/>- Closer를 만들 때<br/>- IIFE (즉시 실행 함수 표현식)를 사용할 때<br/>- 조건부로 함수를 정의하거나, runtime에 함수를 선택해야 할 때 | - 간결한 문법이 선호될 때 (특히 한 줄짜리 간단한 함수)<br/>- Callback 함수 내부에서 상위 scope의 `this` context를 유지해야 할 때<br/>(예: `setTimeout`, `setInterval`, 배열 method `forEach`, `map`, `filter` 등, event handler)<br/>- 객체 literal 내에서 method를 정의할 때 `this`가 객체를 가리키게 하고 싶지 않을 때<br/>(또는 lexical `this`가 필요할 때)<br/>- 생성자 함수로 사용하지 않을 함수, `arguments` 객체가 필요 없는 함수 |

```ts
// Optional Parameter
function greet(name: string, age?: number): string {
  return age ? `Hello, ${name} (${age})` : `Hello, ${name}`;
}

// Default Parameter
function pow(base: number, exp: number = 2): number {
  return base ** exp;
}

// Rest Parameter
function sum(...nums: number[]): number {
  return nums.reduce((acc, n) => acc + n, 0);
}

// void
function logMessage(msg: string): void {
  console.log(msg);
}

// type / interface
type MathOp = (a: number, b: number) => number;
const sub: MathOp = (a, b) => a - b;

// Overload
function toArray(x: number): number[];
function toArray(x: string): string[];
function toArray(x: any): any[] {
  return [x];
}
```

- 매개변수와 반환값에 타입을 명시하여 코드의 안전성과 가독성 향상
- 선택적 매개변수는 `?`, 기본값은 `=`, 나머지 매개변수는 `...`로 표현
- 함수 type을 `type` 또는 `interface`로 정의해 재사용 가능

---

# Class

## `function` (ES5)

- 초기 JavaScript에서는 `class` 키워드가 없었고, 함수와 prototype을 사용하여 class와 유사한 기능 구현

```ts
// ES5 이전의 'class' 정의 방식
function Person(name, age) {
  this.name = name;
  this.age = age;
}

// Prototype을 통한 method 추가
Person.prototype.greet = function () {
  return "안녕하세요, " + this.name + "입니다!";
};

// 객체 생성
var person1 = new Person("김철수", 30);
console.log(person1.greet()); // "안녕하세요, 김철수입니다!"

// 상속 구현 (복잡했음)
function Student(name, age, grade) {
  // 부모 생성자 호출
  Person.call(this, name, age);
  this.grade = grade;
}

// Prototype chain 설정 (상속)
Student.prototype = Object.create(Person.prototype);
Student.prototype.constructor = Student;

// Method override
Student.prototype.greet = function () {
  return Person.prototype.greet.call(this) + " " + this.grade + "학년입니다.";
};
```

## `class` (ES6)

- ES6에서 `class` keyword가 도입되어 더 직관적인 객체 지향 문법 제공

```ts
// ES6 클래스 문법
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  greet() {
    return `안녕하세요, ${this.name}입니다!`;
  }
}

// 상속도 간단해짐
class Student extends Person {
  constructor(name, age, grade) {
    super(name, age); // 부모 생성자 호출
    this.grade = grade;
  }

  // Method override
  greet() {
    return `${super.greet()} ${this.grade}학년입니다.`;
  }
}

// 사용 방식은 동일
const student1 = new Student("이영희", 15, 2);
console.log(student1.greet()); // "안녕하세요, 이영희입니다! 2학년입니다."
```

- TypeScript는 JavaScript class에 type system과 추가 기능 제공

```ts
class Person {
  // 속성 type 선언
  name: string;
  private age: number; // 접근 제한자

  constructor(name: string, age: number) {
    this.name = name;
    this.age = age;
  }

  // Method type 선언
  greet(): string {
    return `안녕하세요, ${this.name}입니다!`;
  }

  // getter
  get personAge(): number {
    return this.age;
  }
}
```
