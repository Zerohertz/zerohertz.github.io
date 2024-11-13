---
title: Go (2)
date: 2023-07-17 20:51:00
categories:
- 2. Backend
tags:
- Go
---
# 시작에 앞선 꿀팁

> 빌드와 실행을 동시에?

```shell
$ go run main.go
```

이렇게 실행하면 `go build main.go`와 `./main`을 합쳐서 실행할 수 있다.

<!-- More -->

---

# Collection

## Array

배열의 선언은 `var ${변수명} [${크기}]${자료형}`으로 선언할 수 있다.
또한 `{}`를 통해 초기화를 할 수 있다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	var a [3]int
	a[0] = 1
	fmt.Println("a:", a)

	var b = [3]int{1, 2, 3}
	fmt.Println("b:", b)

	var c = [...]int{1, 2, 3}
	fmt.Println("c:", c)

	var d [2][2]int
	d[0][1] = 7
	fmt.Println("d:", d)

	var e = [2][2]int{
		{1, 2},
		{3, 4},
	}
	fmt.Println("e:", e)
}
```

```shell
$ go run main.go
a: [1 0 0]
b: [1 2 3]
c: [1 2 3]
d: [[0 7] [0 0]]
e: [[1 2] [3 4]]
```

## Slice

하지만 Go에서 배열은 크기를 동적으로 증가할 수 없으며 부분 배열을 발췌할 수 없다.
그래서 Slice라는 자료형이 존재한다.
선언은 `var ${변수명} []${자료형}`으로 할 수 있다.
혹은 `${변수명} := []${자료형}{${초기화}}` 및 `${변수명} := make([]${자료형}, ${길이}, ${용량})`을 사용할 수 있다. (용량에 대해서는 아래서 설명)

```go main.go
package main

import (
	"fmt"
)

func main() {
	var a []int
	a = []int{1, 2, 3}
	fmt.Println("a:", a)

	b := make([]int, 5, 10)
	fmt.Println("b:", b)
	fmt.Println("len(b):", len(b))
	fmt.Println("cap(b):", cap(b))

	var c []int
	if a == nil {
		fmt.Println("a == nil")
	} else {
		fmt.Println("a != nil")
	}
	if c == nil {
		fmt.Println("c == nil")
	} else {
		fmt.Println("c != nil")
	}

	fmt.Println("append(a, 100):", append(a, 100))
	fmt.Println("a:", a)

	var d [][]int
	d = [][]int{
		{1, 2, 3},
		{4, 5, 6},
	}
	fmt.Println("d:", d)
	// d = append(d, [7, 8, 9]) -> Error
	d = append(d, []int{7, 8, 9})
	fmt.Println("append(d, []int{7, 8, 9}):", d)
}
```

여기서 `nil`은 zero value를 의미하는데, `string` 자료형은 예외여서 `""`을 사용해야한다.

```shell
$ go run main.go
a: [1 2 3]
b: [0 0 0 0 0]
len(b): 5
cap(b): 10
a != nil
c == nil
append(a, 100): [1 2 3 100]
a: [1 2 3]
d: [[1 2 3] [4 5 6]]
append(d, []int{7, 8, 9}): [[1 2 3] [4 5 6] [7 8 9]]
```

Slice는 `len`과 `cap`의 개념을 가지고 각각 길이와 용량을 의미한다.
이 개념은 아래 코드를 통해 쉽게 이해할 수 있다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	var s1 []int
	s2 := make([]int, 0, 3)
	fmt.Println(
		"len(s1):", len(s1),
		"\tlen(s2):", len(s2),
		"\tcap(s1):", cap(s1),
		"\tcap(s2):", cap(s2),
	)
	for i := 1; i <= 15; i++ {
		s1 = append(s1, i)
		s2 = append(s2, i)
		fmt.Println(
			"len(s1):", len(s1),
			"\tlen(s2):", len(s2),
			"\tcap(s1):", cap(s1),
			"\tcap(s2):", cap(s2),
		)
	}

	fmt.Println("s1:", s1)
	fmt.Println("s2:", s2)
}
```

용량을 지정하지 않은 slice `s1`과 길이가 `0`, 용량이 `2`인 slice `s2`에 15번 `append()`를 진행한다면?

```shell
$ go run main.go
len(s1): 0 	len(s2): 0 	cap(s1): 0 	cap(s2): 3
len(s1): 1 	len(s2): 1 	cap(s1): 1 	cap(s2): 3
len(s1): 2 	len(s2): 2 	cap(s1): 2 	cap(s2): 3
len(s1): 3 	len(s2): 3 	cap(s1): 4 	cap(s2): 3
len(s1): 4 	len(s2): 4 	cap(s1): 4 	cap(s2): 6
len(s1): 5 	len(s2): 5 	cap(s1): 8 	cap(s2): 6
len(s1): 6 	len(s2): 6 	cap(s1): 8 	cap(s2): 6
len(s1): 7 	len(s2): 7 	cap(s1): 8 	cap(s2): 12
len(s1): 8 	len(s2): 8 	cap(s1): 8 	cap(s2): 12
len(s1): 9 	len(s2): 9 	cap(s1): 16 	cap(s2): 12
len(s1): 10 	len(s2): 10 	cap(s1): 16 	cap(s2): 12
len(s1): 11 	len(s2): 11 	cap(s1): 16 	cap(s2): 12
len(s1): 12 	len(s2): 12 	cap(s1): 16 	cap(s2): 12
len(s1): 13 	len(s2): 13 	cap(s1): 16 	cap(s2): 24
len(s1): 14 	len(s2): 14 	cap(s1): 16 	cap(s2): 24
len(s1): 15 	len(s2): 15 	cap(s1): 16 	cap(s2): 24
s1: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
s2: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
```

길이의 제한을 받지 않고 늘어나는 순간마다 `len()`이 `s1`, `s2` 모두 증가하는 것을 확인할 수 있다.
`s1`은 첫 용량이 0이기 때문에 첫 `append()`에서 1으로 증가하고 두 번째 `append()`에서는 1의 두 배인 2, 세 번째 `append()`에서는 2의 두 배인 4인 것을 확인할 수 있다.
Slice는 용량을 초과하는 경우 현재 용량의 2배에 해당하는 용량을 새로운 underlying array를 생성하고 기존 배열 값들을 모두 새 배열에 복제하고 다시 슬라이스를 할당한다.
따라서 `s2`는 첫 용량이 3이기 때문에 첫 용량 초과에 용량이 6으로 증가한 것을 확인할 수 있다.

<details>
<summary>
Underlying array가 뭐야?
</summary>

Golang에서 슬라이스 (slice)는 underlying array (기본 배열)을 가리키는 포인터를 포함하고 있습니다. 이 underlying array는 실제 데이터가 저장되는 공간입니다.
슬라이스는 길이 (length)와 용량 (capacity) 정보와 함께 underlying array를 가리키는 포인터를 갖습니다. 슬라이스를 생성할 때, 기본 배열은 자동으로 할당되며, 슬라이스의 요소들은 이 기본 배열에 저장됩니다. 슬라이스는 해당 배열의 일부분을 가리키므로, 배열의 변경은 슬라이스에도 영향을 줍니다.
슬라이스의 일부를 수정하면 underlying array의 해당 부분도 수정됩니다. 마찬가지로, 다른 슬라이스를 생성하고 해당 슬라이스의 요소를 변경하면, 같은 underlying array를 공유하는 다른 슬라이스들도 영향을 받습니다.
예를 들어, 다음 코드를 살펴보겠습니다:

```go
package main

import "fmt"

func main() {
	array := [5]int{1, 2, 3, 4, 5}
	slice := array[1:3]
	fmt.Println("Array:", array)
	fmt.Println("Slice:", slice)

	slice[0] = 10
	fmt.Println("Array:", array)
	fmt.Println("Slice:", slice)
}
```

위의 코드에서 array는 길이가 5인 정수형 배열이고, slice는 array의 일부분인 슬라이스입니다. 슬라이스 slice는 array의 인덱스 1부터 2까지의 요소를 가리키고 있습니다. 결과는 다음과 같을 것입니다:

```bash
Array: [1 2 3 4 5]
Slice: [2 3]
Array: [1 10 3 4 5]
Slice: [10 3]
```

슬라이스 slice의 첫 번째 요소를 10으로 변경한 후, array와 slice를 출력하면, array의 해당 부분도 변경된 것을 확인할 수 있습니다. 이는 slice가 array의 underlying array를 공유하기 때문입니다.
따라서, underlying array는 슬라이스의 데이터를 저장하는 실제 공간이며, 슬라이스의 요소들은 해당 배열에 저장됩니다. 슬라이스를 통해 배열을 조작하면 underlying array도 함께 변경됩니다.

</details>
<br />

Python과 유사하게 slice를 복사할 때 단순히 `:=`를 사용하면 shallow copy된다.
Deep copy를 하려면 `copy()` 함수를 아래와 같이 사용해야한다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	s := []int{1, 2, 3}
	fmt.Println("s:", s)
	shallow := s
	shallow[1] = 100
	fmt.Println("s:", s, "\tshallow:", shallow)

	// var deep []int -> deep: []
	deep := make([]int, len(s), cap(s))
	copy(deep, s)
	fmt.Println("deep:", deep)

	s[2] = 1000
	fmt.Println("s:", s, "\tdeep:", deep)
}
```

```shell
$ go run main.go
s: [1 2 3]
s: [1 100 3] 	shallow: [1 100 3]
deep: [1 100 3]
s: [1 100 1000] 	deep: [1 100 3]
```

## Map

Map은 hash table을 구현한 자료 구조로 python의 dictionary와 유사하다.
선언은 `var ${변수명} map[${Key_자료형}]${Value_자료형}`와 같이 할 수 있다.
하지만 이 상태 (nil map)에서는 어떤 값도 쓸 수 없기 때문에 `make(map[${Key_자료형}]${Value_자료형})`로 변수를 초기화해야 사용할 수 있다.
혹은 `${변수명} = map[${Key_자료형}]${Value_자료형}{}`와 같이 선언하면 초기화를 동시에 할 수 있다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	var m1 map[string]string
	fmt.Println("m1:", m1)
	// m1["asdf"] = "zxcv" -> Error
	m1 = make(map[string]string)
	m1["asdf"] = "zxcv"
	m1["qw"] = "er"
	fmt.Println("m1:", m1)

	m2 := map[int]int{}
	fmt.Println("m2:", m2)
	m2[2] = 71
	m2[3] = 14
	fmt.Println("m2:", m2)
}
```

```shell
$ go run main.go
m1: map[]
m1: map[asdf:zxcv qw:er]
m2: map[]
m2: map[2:71 3:14]
```

`val := map[key]` 혹은 `val, exists := map[key]` 와 같이 map을 사용할 수 있다.
후자와 같이 사용 시 `exists`에 `key`가 존재하는지 여부를 저장한다.
또한 map은 반복문에서 `for k, v := range m`와 같이 사용하면 python dictionary의 `items()`와 유사하게 사용할 수 있다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	m := map[string]string{
		"Zero": "Hertz",
		"오":    "효근",
		"먼지":   "cat",
	}
	val, exists := m["cat"]
	if exists {
		fmt.Println("m[\"cat\"]:", val, exists)
	} else {
		fmt.Println("m[\"cat\"]:", val, exists)
	}
	for key, val := range m {
		fmt.Println(key, val)
	}
}
```

```shell
$ go run main.go
m["cat"]:  false
Zero Hertz
오 효근
먼지 cat
```

---

# Function

Go에서는 기본적으로 `func ${함수명}(${변수명} ${자료형}) ${자료형}`와 같이 함수를 정의할 수 있다.

```go main.go
package main

import "fmt"

func main() {
	test("Hello, World!")
}

func test(msg string) {
	fmt.Println(msg)
}
```

## Pass By Reference

함수의 입력은 두 가지 방법으로 정의할 수 있다.

+ Pass By Value
  + 함수에 변수 전달 시 해당 변수의 값이 복사되어 함수의 매개변수로 전달
  + 함수 내 매개변수 수정 시 원본 변수 영향 X
+ Pass By Reference
  + 함수에 변수 전달 시 해당 변수의 메모리 주소를 함수의 매개변수로 전달
  + 함수 내 매개변수 수정 시 원본 변수 영향 O

아래 예제를 통해 차이를 알 수 있다.

```go main.go
package main

import "fmt"

func main() {
	var mv, mp string
	mv = "Hello, World!"
	mp = "Hello, World!"
	test(mv, &mp)
	fmt.Println("main():\t", mv)
	fmt.Println("main():\t", mp)
}

func test(msg_var string, msg_pt *string) {
	fmt.Println("test():\t", msg_var)
	fmt.Println("test():\t", *msg_pt)
	msg_var = "Changed"
	*msg_pt = "Changed"
	fmt.Println("test():\t", msg_var)
	fmt.Println("test():\t", *msg_pt)
}
```

```shell
$ go run main.go
test():  Hello, World!
test():  Hello, World!
test():  Changed
test():  Changed
main():  Hello, World!
main():  Changed
```

## Variadic Function

가변 파라미터를 함수에 전달할 때 `...`을 사용한다.
하지만 `...`은 마지막 파라미터에서만 사용한다.
즉, `(x ...int, y ...string)`과 같이 사용할 수 없다.
아래 예제에서는 단일 문자열을 전달할 수도 있고 여러 문자열을 전달할 수도 있다.

```go main.go
package main

import "fmt"

func main() {
	test("A", "B", "C")
}

func test(msg ...string) {
	fmt.Println(msg)
	for idx, tmp := range msg {
		fmt.Println(idx, tmp)
	}
}
```

```shell
$ go run main.go
[A B C]
0 A
1 B
2 C
```

## Return

함수의 출력이 2개 이상인 경우 `func ${함수명}(...)` 뒤에 `(${자료형}, ${자료형}, ...)`과 같이 정의할 수 있다.
혹은 `(${변수명} ${자료형}, ${변수명} ${자료형}, ...)`과 같이 정의할 수 있다.
아래 예제에서 첫 번째와 같이 정의한 함수가 `sum()` 함수이고, 두 번째와 같이 정의한 함수가 `sum2()`이다.
`sum2()`와 같이 함수를 정의할 때 `return` 뒤에 별도의 변수가 존재하지 않더라도 꼭 `return`을 명시해야 한다.
결과는 동일한 것을 확인할 수 있다.

```go main.go
package main

import "fmt"

func main() {
	count, total := sum(1, 5, 3)
	fmt.Println(count, total)
	count2, total2 := sum2(1, 5, 3)
	fmt.Println(count2, total2)
}

func sum(nums ...int) (int, int) {
	total := 0
	count := 0
	for _, n := range nums {
		total += n
		count++
	}
	return count, total
}

func sum2(nums ...int) (count int, total int) {
	for _, n := range nums {
		total += n
	}
	count = len(nums)
	return
}
```

```shell
$ go run main.go
3 9
3 9
```