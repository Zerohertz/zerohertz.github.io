---
title: Go (1)
date: 2023-07-03 22:15:57
categories:
- 2. Backend
tags:
- Go
---
# Introduction

> [Go](https://go.dev/)는 구글에서 개발한 프로그래밍 언어로, 간결하고 효율적인 코드 작성을 위해 설계된 언어입니다. Go는 정적 타입 언어로, C와 유닉스 환경에서 사용되는 도구들의 장점을 결합한 언어로 개발되었습니다.

## Features of Go

1. 간결하고 가독성이 좋은 문법
   + C 스타일의 문법을 가지고 있으며, 불필요한 기능을 제거하여 코드를 간결하게 작성 가능
   + 문법이 간단하고 가독성이 좋아 새로운 개발자들의 빠른 숙달
2. 효율적인 동시성 지원
   + 동시성을 위한 기능 내장 $\rightarrow$ 병렬 프로그래밍의 쉬운 구현
   + Goroutine은 경량 스레드로, 작은 메모리 오버헤드로 수많은 고루틴을 동시에 실행 가능 $\rightarrow$ 대규모 시스템에서 효율적인 동시성 처리 구현 가능
3. 빠른 컴파일과 실행 속도
   + 정적 타입 언어지만 빠른 컴파일 속도
   + Garbage collection을 통한 편한 메모리 관리 $\rightarrow$ 대규모 시스템에서도 빠른 실행 속도를 제공합니다.
4. 강력한 표준 라이브러리
   + 풍부한 표준 라이브러리를 제공 (네트워킹, 암호화, 웹 개발, 데이터베이스 액세스 등 다양한 작업)
5. 크로스 플랫폼 지원
   + 다양한 운영 체제와 아키텍처에서 사용할 수 있도록 크로스 플랫폼 지원 제공
   + 하나의 코드베이스로 여러 플랫폼에서 실행할 수 있는 이식성 높은 프로그램 개발 가능

Go는 웹 서버, 분산 시스템, 클라우드 서비스, 네트워크 프로그래밍 등 다양한 분야에서 사용된다.
특히, 동시성 처리에 강점을 가지고 있어 대규모 시스템의 성능을 향상시킬 수 있다.
또한, Go는 개발자들의 생산성을 높이기 위해 설계된 언어로, 간결한 문법과 강력한 도구들을 제공하여 개발자들이 효율적으로 코드를 작성할 수 있도록 도와준다.

<!-- More -->

## Install & Setup on MacOS (Apple Silicon)

```shell Install
$ brew install go
$ go version
go version go1.20.5 darwin/arm64
```

```shell Setup
$ vi ~/.zshrc
export GOROOT=${YOUR_GOROOT}
export PATH=$GOROOT/bin:$PATH
$ source ~/.zshrc
$ go env | grep GOROOT
GOROOT=${YOUR_GOROOT}
```

+ `GOPATH`: Go에서 사용되는 환경 변수
  + 소스 코드와 컴파일된 바이너리 파일을 관리하기 위해 일정한 디렉토리 구조 필요 $\rightarrow$ `GOPATH`로 설정!
  + 패키지들을 가져오고 설치하는 위치를 지정
+ 기본 설정
  + Linux 및 MacOS: `$HOME/go`
  + Windows: `%USERPROFILE%\go`
+ `GOPATH` 설정 시
  + Go 패키지들이 해당 경로 아래 디렉토리에 설치
  + Go 컴파일러가 해당 경로를 검색하여 패키지 모색

```shell
~/go $ tree -L 2
.
├── bin
│   ├── dlv
│   ├── go-outline
│   ├── goimports
│   ├── gomodifytags
│   ├── goplay
│   ├── gopls
│   ├── gotests
│   ├── impl
│   └── staticcheck
└── pkg
    ├── mod
    └── sumdb
```

## Install & Setup on Ubuntu

```shell
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install wget
$ wget https://golang.org/dl/go1.21.3.linux-amd64.tar.gz
$ rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.21.3.linux-amd64.tar.gz
$ vi ~/.zshrc
export GOROOT=/usr/local/go
export PATH=$GOROOT/bin:$PATH
$ source ~/.zshrc
$ go version
go version go1.21.3 linux/amd64
```

## Hello, World!

이렇게 설치와 설정을 완료했으니,,, `Hello, World!`를 해보자!

```go main.go
package main

import "fmt"

func main() {
	fmt.Print("Hello, World!")
}
```

+ `package main`
  + Go에서 실행 가능한 프로그램의 진입점을 정의하는 패키지 선언
  + `main()` 함수가 포함되어야 하며 프로그램의 실행 흐름은 이 함수에서 시작

```shell Build
$ go build main.go
$ ls | grep main
main
main.go
$ ./main
Hello, World!
```

MacOS에서 zsh를 이용하는 `%`가 추가되어 출력되는 경우 `~/.zshrc`에 `export PROMPT_EOL_MARK=`를 추가하면 된다.

---

# Basic Grammar

## Const & Variable

> `const ${변수명} ${자료형} = ${값}`

```go main.go
package main

import "fmt"

func main() {
	const a bool = true
	fmt.Println(!a)
	const b int = 100
	fmt.Println(b)
	const c float64 = 3.14
	fmt.Println(c)
	const d string = "안녕하세요, 오효근입니다."
	fmt.Println(d)
}
```

```shell
$ go build main.go
$ ./main
false
100
3.14
안녕하세요, 오효근입니다.
```

> `var ${변수명} ${자료형} = ${값}`

```go main.go
package main

import "fmt"

func main() {
	var a bool = true
	fmt.Println(!a)
	var b int = 100
	fmt.Println(b)
	var c float64 = 3.14
	fmt.Println(c)
	var d string = "안녕하세요, 오효근입니다."
	fmt.Println(d)
}
```

```shell
$ go build main.go
$ ./main
false
100
3.14
안녕하세요, 오효근입니다.
```

## Operator

|Type|Operator|Mean|
|:-:|:-:|:-:|
|산술|+|덧셈|
|산술|-|뺄셈|
|산술|*|곱셈|
|산술|/|나눗셈|
|산술|%|나머지|
|비교|==|값이 같은지 비교|
|비교|!=|값이 다른지 비교|
|비교|<|미만|
|비교|>|초과|
|비교|<=|이하|
|비교|>=|이상|
|논리|&&|논리 AND|
|논리|\|\||논리 OR|
|논리|!|논리 NOT|
|할당|=|값 할당|
|할당|+=|덧셈 후 할당|
|할당|-=|뺄셈 후 할당|
|할당|*=|곱셈 후 할당|
|할당|/=|나눗셈 후 할당|
|할당|%=|나머지 연산 후 할당|
|증감|++|1 증가|
|증감|--|1 감소|
|비트|&|비트 AND|
|비트|\||비트 OR|
|비트|^|비트 XOR|
|비트|<<|비트 왼쪽 시프트|
|비트|>>|비트 오른쪽 시프트|
|비트|&^|비트 AND NOT|
|포인터|*|포인터 역참조|
|포인터|&|변수의 주소|
|기타|.|구조체 필드 접근|
|기타|()|함수 호출|
|기타|[]|배열, 슬라이스, 맵 인덱스 접근|
|기타|:|맵에 값 할당 및 슬라이스 범위 지정|

## Data Type & Type Conversion

+ Boolean: `bool`
+ Numeric
  + Integer: `int`, `int8`, `int16`, `int32`, `int64`
  + Unsigned Integer: `uint`, `uint8`, `uint16`, `uint32`, `uint64`
  + Float: `float32`, `float64`
  + Complex: `complex64`, `complex128`
+ String: `string`

Go에서 `string`은 참 독특한 성격을 지닌다.
`''`을 사용하면 단일 문자만 변수에 저장할 수 있으며 아래와 같이 `int32`로 지정되어 `""`을 사용해야한다.

```go main.go
package main

import (
	"fmt"
	"reflect"
)

func main() {
	a := "test"
	// b := 'test' -> Error
	b := 't'
	fmt.Println(reflect.TypeOf(a))
	fmt.Println(reflect.TypeOf(b))
}
```

```shell
$ go build main.go
$ ./main
string
int32
```

심지어 `string`인 변수는 바꿀 수 없다.
슬라이싱으로 특정 문자를 바꾸려면 아래와 같이 진행해야한다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	var S string = "asdfqwer"
	var L int = len(S)
	S = S[0:2] + "D" + S[3:L]
	fmt.Print(S)
}
```

```shell
$ go build main.go
$ ./main
asDfqwer
```

또 Go에서는 항상 명시적으로 형 변환을 진행해야하는데 `string`은 아래와 같이 진행한다.

```go main.go
package main

import (
	"fmt"
	"reflect"
	"strconv"
)

func main() {
	var A int = 10
	var B float32 = 3.14
	var C string = "1234.56"

	var D float32 = float32(A)
	var E float64 = float64(A)
	F := string(A)

	var G int = int(B)
	var H string = strconv.Itoa(int(B))
	var I string = strconv.FormatFloat(float64(B), 'f', -1, 32)

	J, _ := strconv.Atoi(C)
	K, _ := strconv.ParseFloat(C, 32)

	fmt.Println("A:", reflect.TypeOf(A), A)
	fmt.Println("B:", reflect.TypeOf(B), B)
	fmt.Println("C:", reflect.TypeOf(C), C)
	fmt.Println("D:", reflect.TypeOf(D), D)
	fmt.Println("E:", reflect.TypeOf(E), E)
	fmt.Println("F:", reflect.TypeOf(F), F)
	fmt.Println("G:", reflect.TypeOf(G), G)
	fmt.Println("H:", reflect.TypeOf(H), H)
	fmt.Println("I:", reflect.TypeOf(I), I)
	fmt.Println("J:", reflect.TypeOf(J), J)
	fmt.Println("K:", reflect.TypeOf(K), K)
}
```

```shell
$ go build main.go
$ ./main
A: int 10
B: float32 3.14
C: string 1234.56
D: float32 10
E: float64 10
F: string 

G: int 3
H: string 3
I: string 3.14
J: int 0
K: float64 1234.56005859375
```

정리하자면 문자열을 정수로 변환하려면 아래와 같다.

+ `string` $\rightarrow$ `int`: `strconv.Atoi()`
+ `string` $\rightarrow$ `float64`: `strconv.ParseFloat()`
+ `int` $\rightarrow$ `string`: `strconv.Itoa()`
+ `float64` $\rightarrow$ `string`: `strconv.FormatFloat()`

## Conditional Statement

Go에서 조건문을 사용할 때 `else if` 혹은 `else`가 중괄호 `} {` 사이에 존재해야하고 같은 라인에 있어야한다.

```go main.go
package main

import (
	"fmt"
)

func main() {
	var A int = 10
	if A == 10 {
		fmt.Println("Hi")
	} else {
		fmt.Println("Bye")
	}
	A = 20
	if A == 10 {
		fmt.Println("Hi")
	} else if A == 12 {
		fmt.Println("Hi")
	} else {
		fmt.Println("Bye")
	}
}
```

```shell
$ go build main.go
$ ./main
Hi
Bye
```

## Loop Statement

반복문은 C와 매우 유사하여 쉽다!

```go main.go
package main

import (
	"fmt"
)

func main() {
	for i := 0; i < 10; i++ {
		fmt.Println(i, "Hi!")
	}
}
```

```shell
$ go build main.go
$ ./main
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

# For Algorithm

입출력 시 `fmt.Print` 및 `fmt.Scan`을 사용하면 시간 초과가 발생할 수 있다.
따라서 아래와 같이 입출력을 사용해야 해결할 수 있다.

```go
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	var reader *bufio.Reader = bufio.NewReader(os.Stdin)
	var writer *bufio.Writer = bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	var N int
	fmt.Fscanln(reader, &N)
	fmt.Fprintln(writer, "Hello, World")
}
```