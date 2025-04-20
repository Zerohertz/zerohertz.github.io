---
title: JavaScript & TypeScript (1)
date: 2025-04-20 12:08:00
categories:
  - Etc
tags:
  - JavaScript
  - TypeScript
---

# Introduction

| Feature        | JavaScript (JS)       | TypeScript (TS)               |
| -------------- | --------------------- | ----------------------------- |
| Type System    | Dynamic               | Static + Dynamic              |
| Compilation    | Interpreted (runtime) | Compiled to JS (transpile)    |
| Type Checking  | No                    | Yes (compile-time)            |
| IDE Support    | Basic                 | Advanced (IntelliSense, etc.) |
| Learning Curve | Easy                  | Slightly higher               |
| Community      | Very large            | Large, growing                |
| Ecosystem      | Huge                  | Uses JS ecosystem             |
| Error Catching | Runtime               | Compile-time + Runtime        |
| Annotation     | Not required          | Optional (but recommended)    |
| OOP Support    | Prototype-based       | Class-based (ES6+), Interface |

```shell Install.sh
$ npm install -g typescript
$ npm init -y
Wrote to /path/to/package.json:
...
$ tsc --init
Created a new tsconfig.json with:
  target: es2016
  module: commonjs
  strict: true
  esModuleInterop: true
  skipLibCheck: true
  forceConsistentCasingInFileNames: true
You can learn more at https://aka.ms/tsconfig
```

<!-- More -->

JavaScript는 web, server, app 등 거의 모든 곳에서 사용되는 universal language다.
하지만 type이 없어서 대규모 project에서 아래와 같은 문제가 발생한다.

- Type Error: 변수, 함수의 type이 명확하지 않아 runtime error가 자주 발생
- IDE Support 부족: code 자동완성, refactoring, navigation 등에서 한계
- code 가독성 저하: type이 명확하지 않아 협업/유지보수 시 어려움

TypeScript는 이런 문제를 해결하기 위해 등장했다. 주요 장점은 아래와 같다.

- Type Safety: compile 단계에서 type error를 미리 잡아준다
- Better IDE Support: IntelliSense, 자동완성, type 추론 등 개발 생산성 향상
- Refactoring 용이: type 정보 기반으로 안전하게 code 변경 가능
- 대규모 project에 적합: 명확한 interface, type alias 등으로 협업/유지보수에 강점

{% cq %}
즉, TypeScript는 JavaScript의 superset으로, 기존 JS code를 그대로 사용하면서 type을 추가해 더 안전하고 생산적인 개발을 가능하게 해준다.
{% endcq %}

---

# Hello, World

```ts src/index.ts
function greeter(person: string) {
  return "Hello, " + person;
}

let user = "World!";

console.log(greeter(user));
```

```shell
$ tsc src/index.ts
$ ls src
index.js        index.ts
$ node src/index.js
Hello, World!
```

위 code는 TypeScript의 기본적인 문법과 compile 과정을 보여준다.

- `function greeter(person: string)`
  - 함수의 parameter에 `string` type을 명시적으로 지정 (type annotation)
  - 만약 다른 type (ex. number 등)을 넣으면 compile 단계에서 error 발생
- `let user = "World!";`
  - 변수 선언 시 type을 명시하지 않아도, TypeScript가 자동으로 type을 추론 (type inference)
- `console.log(greeter(user));`
  - JavaScript와 동일하게 동작

TypeScript file(`.ts`)은 직접 실행할 수 없고, 반드시 JavaScript(`.js`)로 compile(transpile)해야 한다.

- `tsc src/index.ts` : TypeScript compiler(tsc)로 `.ts` file을 `.js`로 변환
- `node src/index.js` : 변환된 JavaScript file을 실행

{% cq %}
즉, TypeScript는 기존 JavaScript 개발 flow에 type system만 추가된 형태라고 볼 수 있다. 기존 JS code를 그대로 사용하면서, type을 명확히 하여 더 안전한 code를 작성할 수 있다.
{% endcq %}

```ts src/index.ts
function greeter(person: string) {
  return "Hello, " + person;
}

let user = [1, 2, 3];

console.log(greeter(user));
```

```shell
$ tsc src/index.ts
src/index.ts:7:21 - error TS2345: Argument of type 'number[]' is not assignable to parameter of type 'string'.
7 console.log(greeter(user));
                      ~~~~
Found 1 error in src/index.ts:7
```

위 예제는 TypeScript의 type checking이 어떻게 동작하는지 보여준다.

- `greeter` 함수는 parameter로 `string` type만 받도록 선언되어 있다.
- `user` 변수는 `number[]` (number array)로 선언되어 있음에도 불구하고, 함수에 그대로 전달하면 compile 단계에서 error가 발생한다.
- JavaScript에서는 이런 type mismatch가 runtime에야 발견되지만, TypeScript는 compile 시점에 미리 error를 알려준다.

{% cq %}
즉, TypeScript의 가장 큰 장점은 잘못된 type 사용을 미리 방지해주기 때문에, 대규모 project에서 bug를 줄이고 code의 신뢰성을 높일 수 있다는 점이다.
{% endcq %}

하지만 위 code는 compile이 진행되어 `src/index.js`로 출력됨을 확인할 수 있고, 심지어 실행도 된다.

```js src/index.js
function greeter(person) {
  return "Hello, " + person;
}
var user = [1, 2, 3];
console.log(greeter(user));
```

```shell
$ node src/index.js
Hello, 1,2,3
```

이는 TypeScript가 기본적으로 compile error를 표시하지만, JavaScript 생성은 막지 않기 때문이다. 이런 동작은 `tsconfig.json`에서 아래와 같이 조절할 수 있다.

```json tsconfig.json
{
  "compilerOptions": {
    ...
    "noEmitOnError": true,  // compile error가 발생하면 JS file을 생성하지 않음
    ...
  }
}
```

하지만 `tsconfig.json`을 수정했는데도 compile error가 발생해도 JS file이 생성되는 경우가 있다. 이는 보통 아래와 같은 이유 때문이다:

1. `tsc` 명령어에 특정 file을 직접 지정한 경우: `tsc src/index.ts`처럼 특정 file을 직접 지정하면 `tsconfig.json`의 설정이 무시될 수 있다. 대신 `tsc`만 실행하여 project 전체에 설정을 적용해야 한다.
2. 설정 file이 제대로 적용되지 않은 경우: 설정 file이 있는 directory에서 명령을 실행해야 한다.

```shell
# 올바른 방법 (tsconfig.json 설정 적용)
$ tsc

# 아래처럼 특정 file을 지정하면 tsconfig.json이 무시될 수 있음
$ tsc src/index.ts
```

또는 특정 file에 설정을 적용하고 싶다면:

```shell
tsc --noEmitOnError src/index.ts
```

위와 같이 명령어에 option을 직접 추가해 사용할 수도 있다.

---

# Const & Variable

TypeScript에서는 JavaScript의 변수 선언 방식을 그대로 사용하면서, type system을 추가하여 더 안전한 code 작성이 가능하다.

| Keyword | 재할당    | 재선언    | Hoisting       | Scope       | 특징                                |
| ------- | --------- | --------- | -------------- | ----------- | ----------------------------------- |
| `var`   | ✅ 가능   | ✅ 가능   | ✅ 변수 선언만 | 함수 scope  | ES5 이전 방식, 현재는 권장하지 않음 |
| `let`   | ✅ 가능   | ❌ 불가능 | ❌ TDZ 적용    | Block scope | 값이 변경되는 변수에 권장           |
| `const` | ❌ 불가능 | ❌ 불가능 | ❌ TDZ 적용    | Block scope | 상수 선언에 권장 (기본 선택)        |

> Hoisting: JavaScript에서 code가 실행되기 전에 변수나 함수의 선언문을 해당 범위의 맨 위로 끌어올리는 현상

```ts
// var (권장하지 않음)
var count = 10;
var count = 20; // 재선언 가능
count = 30; // 재할당 가능

// let
let score = 100;
// let score = 200; // Error: 같은 scope에서 재선언 불가
score = 200; // 재할당 가능

// const (권장)
const PI = 3.14;
// PI = 3.15;       // Error: 재할당 불가
// const PI = 3.15; // Error: 재선언 불가
```

{% cq %}
TypeScript에서는 기본적으로 `const`를 사용하고, 필요한 경우에만 `let`을 사용하는 것이 권장된다.
`var`는 가급적 사용하지 않는 것이 좋다.
{% endcq %}

---

# Operator

| Type |     Operator      |                 Mean                  |
| :--: | :---------------: | :-----------------------------------: |
| 산술 |        `+`        |                더하기                 |
| 산술 |        `-`        |                 빼기                  |
| 산술 |        `*`        |                곱하기                 |
| 산술 |        `/`        |                나누기                 |
| 산술 |        `%`        |            나머지 (모듈로)            |
| 산술 |       `++`        |         증가 (전위 또는 후위)         |
| 산술 |       `--`        |         감소 (전위 또는 후위)         |
| 비교 |       `==`        |              동일 (같음)              |
| 비교 |       `!=`        |           다름 (같지 않음)            |
| 비교 |        `>`        |                  큼                   |
| 비교 |        `<`        |                 작음                  |
| 비교 |       `>=`        |              크거나 같음              |
| 비교 |       `<=`        |              작거나 같음              |
| 논리 |       `&&`        |               논리 AND                |
| 논리 |      `\|\|`       |                논리 OR                |
| 논리 |        `!`        |               논리 NOT                |
| 비트 |        `&`        |               비트 AND                |
| 비트 |       `\|`        |                비트 OR                |
| 비트 |        `^`        |         비트 XOR (배타적 OR)          |
| 비트 |        `~`        |               비트 NOT                |
| 비트 |       `<<`        |              왼쪽 시프트              |
| 비트 |       `>>`        |       오른쪽 시프트 (부호 유지)       |
| 비트 |       `>>>`       | 오른쪽 시프트 (부호 무시, 0으로 채움) |
| 대입 |        `=`        |                 할당                  |
| 대입 |       `+=`        |              더하고 할당              |
| 대입 |       `-=`        |               빼고 할당               |
| 대입 |       `*=`        |              곱하고 할당              |
| 대입 |       `/=`        |              나누고 할당              |
| 대입 |       `%=`        |          나머지 값으로 할당           |
| 대입 |       `&=`        |           비트 AND 후 할당            |
| 대입 |       `\|=`       |            비트 OR 후 할당            |
| 대입 |       `^=`        |           비트 XOR 후 할당            |
| 대입 |       `<<=`       |          왼쪽 시프트 후 할당          |
| 대입 |       `>>=`       |         오른쪽 시프트 후 할당         |
| 대입 |      `>>>=`       |   오른쪽 시프트 (부호 무시) 후 할당   |
| 특수 |       `?:`        |       삼항 연산자 (조건 연산자)       |
| 특수 |   `instanceof`    |            객체 타입 확인             |
| 특수 | `()`, 타입 캐스팅 |                형 변환                |

---

# Types

## Data Type

TypeScript는 JavaScript의 모든 data type을 지원하면서 추가적인 type 안전성을 제공한다.

| Data Type   | 설명                                             | 예시                                             |
| ----------- | ------------------------------------------------ | ------------------------------------------------ |
| `number`    | 모든 숫자(정수, 소수, NaN, Infinity 등)          | `let n: number = 42;`                            |
| `string`    | 문자열(작은따옴표, 큰따옴표, 백틱 사용)          | `let s: string = "hello";`                       |
| `boolean`   | 논리값(true/false)                               | `let b: boolean = true;`                         |
| `object`    | JavaScript object(key-value pair)                | `let o: object = { name: "Kim" };`               |
| `array`     | 동일 type을 가진 값들의 순서 있는 collection     | `let arr: number[] = [1, 2, 3];`                 |
| `tuple`     | 고정된 길이와 각 위치별 정해진 type을 가진 array | `let t: [string, number] = ["Kim", 30];`         |
| `enum`      | 명명된 상수 집합                                 | `enum Color { Red, Green, Blue }`                |
| `any`       | 모든 type이 허용됨(type 검사 미적용)             | `let a: any = 4;` `a = "string";`                |
| `unknown`   | 모든 type이 허용되지만, 사용 전 type 확인 필요   | `let u: unknown = 4;`                            |
| `void`      | 값을 반환하지 않는 함수의 반환 type              | `function log(): void { console.log("Hi"); }`    |
| `never`     | 절대 발생하지 않는 값의 type                     | `function error(): never { throw new Error(); }` |
| `null`      | 의도적으로 값이 없음을 나타냄                    | `let n: null = null;`                            |
| `undefined` | 값이 할당되지 않은 상태                          | `let u: undefined = undefined;`                  |
| `literal`   | 특정 값만 허용하는 type                          | `let direction: "left" \| "right";`              |

```ts
// Union Type: 여러 type 중 하나를 허용
let id: string | number;
id = 101; // OK
id = "A101"; // OK
// id = true;  // Error: boolean은 허용되지 않음

// Intersection Type: 여러 type을 모두 만족
type Employee = { name: string; id: number };
type Manager = { supervises: string[] };
type ManagerEmployee = Employee & Manager;
// 반드시 name, id, supervises 속성이 모두 필요함
const manager: ManagerEmployee = {
  name: "Kim",
  id: 123,
  supervises: ["Lee", "Park"],
};

// Type Alias: 복잡한 type에 이름 부여
type UserID = string | number;
type Point = { x: number; y: number };

// Union type과 literal type을 조합한 예
type Status = "pending" | "approved" | "rejected";
let currentStatus: Status = "pending";
// currentStatus = "waiting";  // Error: "waiting"은 Status type에 없음
```

{% cq %}
TypeScript의 강력한 type system을 활용하면 compile 단계에서 잠재적 오류를 미리 발견할 수 있어 코드의 안정성이 크게 향상된다. 필요에 따라 strict하게 또는 유연하게 type을 활용하는 것이 중요하다.
{% endcq %}

## Type Conversion

TypeScript에서는 다양한 방법으로 type 변환이 가능하다.

### 암시적 변환 (Implicit Conversion)

JavaScript처럼 TypeScript도 일부 상황에서 암시적 type 변환을 수행한다:

```ts
// String concatenation에서 암시적 변환
let num = 42;
let message = "Answer: " + num; // num이 string으로 암시적 변환됨
console.log(message); // "Answer: 42"

// 비교 연산자에서의 암시적 변환
console.log("42" == 42); // true (값만 비교, type은 무시)
console.log("42" === 42); // false (값과 type 모두 비교)
```

### 명시적 변환 (Explicit Conversion)

TypeScript에서 명시적으로 type을 변환하는 방법:

| 변환 대상           | 변환 방법                                | 예시                                                                                                           |
| ------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `string` → `number` | `Number()`, `parseInt()`, `parseFloat()` | `const num = Number("42");`<br/>`const int = parseInt("42px");` → `42`<br/>`const float = parseFloat("3.14");` |
| `number` → `string` | `String()`, `.toString()`                | `const str1 = String(42);`<br/>`const str2 = (42).toString();`                                                 |
| 값 → `boolean`      | `Boolean()`                              | `const bool = Boolean("");` → `false`<br/>`const bool2 = Boolean(1);` → `true`                                 |
| Type assertion      | `as` keyword, `<>` 문법                  | `const len = (value as string).length;`<br/>`const len2 = (<string>value).length;`                             |

```ts
// 숫자 → 문자열 변환
let num = 42;
let strNum1 = String(num); // "42"
let strNum2 = num.toString(); // "42"
let strNum3 = `${num}`; // template literal 사용 (권장)

// 문자열 → 숫자 변환
let str = "42";
let numStr1 = Number(str); // 42
let numStr2 = parseInt(str); // 42
let numStr3 = parseFloat("3.14"); // 3.14

// Boolean 변환
let truthy = Boolean(1); // true
let falsy = Boolean(0); // false

// 특수한 경우
console.log(Number("42px")); // NaN (숫자로 변환 불가)
console.log(parseInt("42px")); // 42 (숫자로 시작하는 부분만 parsing)
```

### Type Assertion (Type 단언)

TypeScript에서 특정 값의 type을 compiler에게 명시적으로 알려주는 방법.

```ts
// Type assertion 사용 예시 1: as 구문 사용
let someValue: unknown = "Hello, TypeScript";
let strLength: number = (someValue as string).length;

// Type assertion 사용 예시 2: angle-bracket(<>) 구문 사용
// (JSX와 함께 사용 시 충돌 가능성 있어 권장하지 않음)
let someValue2: unknown = "Another string";
let strLength2: number = (<string>someValue2).length;

// DOM 조작 시 Type Assertion 활용 예
const input = document.getElementById("user-input") as HTMLInputElement;
// 이제 input.value 접근 가능 (HTMLElement에는 value 속성이 없지만 HTMLInputElement에는 있음)
input.value = "New value";

// 복잡한 object Type Assertion
interface User {
  id: number;
  name: string;
  email?: string;
}

const userData: unknown = JSON.parse('{"id": 1, "name": "Kim"}');
const user = userData as User;
console.log(user.name); // "Kim"
```

### `const` Assertion

TypeScript 3.4부터 도입된 `as const`는 object나 array를 deeply immutable하게 만든다.

```ts
// 일반 object는 내부 속성이 수정 가능
const user = {
  id: 101,
  name: "Kim",
};
user.name = "Lee"; // OK

// const assertion을 사용하면 모든 속성이 readonly가 됨
const user2 = {
  id: 101,
  name: "Kim",
} as const;
// user2.name = "Lee"; // Error: Cannot assign to 'name' because it is a read-only property

// Array에도 적용 가능
const countries = ["Korea", "Japan", "China"] as const;
// countries.push("USA"); // Error
// countries[0] = "USA";  // Error

// Literal type 보존에도 유용
const status = {
  SUCCESS: 200,
  ERROR: 500,
} as const;
// status.SUCCESS의 type은 200 (number가 아닌 literal type)
```

{% cq %}
TypeScript에서 type 변환은 필요한 경우에만 명시적으로 수행하는 것이 좋다.
특히 type assertion은 compiler가 check할 수 없는 영역이므로 신중하게 사용해야 한다.
`as const`는 immutability가 필요한 object나 array, 특히 Redux action이나 configuration object 등에서 유용하게 활용할 수 있다.
{% endcq %}

## Type Annotation

TypeScript에서는 변수 선언 시 type을 명시적으로 지정할 수 있다.

```ts
// 기본 type annotation
const name: string = "Kim";
let age: number = 30;
let isActive: boolean = true;
let anyValue: any = "hello"; // 모든 type 허용 (권장하지 않음)

// Array
const numbers: number[] = [1, 2, 3];
const names: Array<string> = ["Kim", "Lee", "Park"];

// Tuple
const tuple: [string, number] = ["Kim", 30];

// object
const user: { name: string; age: number } = { name: "Kim", age: 30 };

// null과 undefined
let empty: null = null;
let notDefined: undefined = undefined;

// Union type (여러 type 중 하나)
let id: string | number = 123;
id = "A123"; // 문자열도 할당 가능
```

## Type Inference

TypeScript는 초기값을 기준으로 type을 자동으로 추론할 수 있어, 항상 type을 명시할 필요는 없다:

```ts
// Type 추론 (명시적 type 선언 없이도 type이 결정됨)
const inferredName = "Kim"; // string으로 추론
let inferredAge = 30; // number로 추론
const inferredActive = true; // boolean으로 추론

// Array도 type 추론
const inferredNumbers = [1, 2, 3]; // number[]로 추론

// Object도 type 추론
const inferredUser = {
  name: "Kim",
  age: 30,
}; // { name: string; age: number }로 추론
```

## `null` vs. `undefined`

| 특성           | `null`                                                              | `undefined`                                                                                                                                                   |
| -------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 의미           | 값이 의도적으로 비어있음을 나타냄                                   | 값이 할당되지 않았음을 나타냄                                                                                                                                 |
| Type           | `typeof null === 'object'`                                          | `typeof undefined === 'undefined'`                                                                                                                            |
| 발생 상황      | - 명시적으로 할당된 경우만 발생                                     | - 변수 선언 후 초기화하지 않은 경우<br/>- object의 존재하지 않는 속성에 접근할 때<br/>- 반환값이 없는 함수의 결과<br/>- 함수의 parameter가 전달되지 않은 경우 |
| JSON 직렬화    | `JSON.stringify({a: null}) === '{"a":null}'`                        | `JSON.stringify({a: undefined}) === '{}'` (속성 자체가 제외됨)                                                                                                |
| 동등 비교      | - `null == undefined` → `true`<br/>- `null === undefined` → `false` | - `undefined == null` → `true`<br/>- `undefined === null` → `false`                                                                                           |
| 기본값 설정    | `const x = null ?? 'default'` → `default`                           | `const x = undefined ?? 'default'` → `default`                                                                                                                |
| 사용 권장 사례 | `object`의 부재를 명시적으로 표현할 때                              | 초기화되지 않은 상태를 나타낼 때                                                                                                                              |

```ts
// strictNullChecks가 true일 때
let name: string;
name = null; // error: null은 string에 할당할 수 없음
name = undefined; // error: undefined는 string에 할당할 수 없음

// 명시적으로 허용하려면
let name: string | null | undefined;
name = null; // 정상
name = undefined; // 정상
```

## `type` vs. `interface` vs. `class`

TypeScript에서 type을 정의하는 세 가지 주요 방법인 `type` alias, `interface`, `class` 간의 차이점을 이해하는 것은 중요하다.
각각의 특징과 사용 사례를 비교해보자.

| 특징             | `type`                                   | `interface`                           | `class`                              |
| ---------------- | ---------------------------------------- | ------------------------------------- | ------------------------------------ |
| 용도             | Type 정의                                | Object 구조 명세                      | Object 생성 blueprint                |
| 확장 방법        | `&` (intersaction)                       | `extends`                             | `extends`                            |
| 선언 병합        | 불가능                                   | 가능 (같은 이름으로 여러번 선언 가능) | 불가능                               |
| 구현             | Type만 정의                              | Type만 정의                           | Type 정의 + 구현 logic               |
| 연산자 사용      | Union (`\|`), intersaction (`&`) 등 가능 | 제한적                                | 불가능                               |
| Primitive/Union  | 모든 type 표현 가능                      | Object 구조만 정의 가능               | Object 구조만 정의 가능              |
| Computed 속성    | 가능                                     | 제한적                                | 불가능                               |
| Runtime          | Compile 시 제거됨                        | Compile 시 제거됨                     | Runtime에 존재                       |
| Instance 생성    | 불가능 (`new` keyword 사용 불가)         | 불가능 (`new` keyword 사용 불가)      | 가능 (`new` keyword로 instance 생성) |
| 적합한 사용 사례 | 복잡한 type, union, intersaction         | API contract, library 정의            | Object instance 생성, OOP 설계       |

```ts
// Type alias
type User = {
  id: number;
  name: string;
};
type Admin = User & {
  role: "admin";
  permissions: string[];
};
// Union type (Interface로는 불가능)
type ID = number | string;

// Interface
interface Vehicle {
  brand: string;
  year: number;
}
interface Car extends Vehicle {
  doors: number;
}
// 선언 병합 (Type으로는 불가능)
interface Vehicle {
  color: string; // 기존 Vehicle에 속성 추가
}

// Class
class Person {
  name: string;
  age: number;

  constructor(name: string, age: number) {
    this.name = name;
    this.age = age;
  }

  greet() {
    return `Hello, my name is ${this.name}`;
  }
}
// Instance 생성 (Type, Interface로는 불가능)
const john = new Person("John", 30);
```

{% cq %}
TypeScript에서는, 상황에 따라 적절한 type 정의 방법을 선택하는 것이 중요하다.
일반적으로 object 구조 명세에는 `interface`를, 복잡한 type 조합에는 `type` alias를, object 생성이 필요할 때는 `class`를 사용하는 것이 좋다.
{% endcq %}
