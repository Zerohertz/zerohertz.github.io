---
title: Queue and Stack
date: 2022-07-21 15:20:14
categories:
- 1. Algorithm
tags:
- Python
---
# Data Structure

+ Definition: Computer science에서 효율적인 접근 및 수정을 위해 자료의 조직, 관리, 저장
  + 데이터 값의 모임, 데이터 간 관계, 데이터에 적용할 수 있는 함수 및 명령
  + 적합한 자료구조 선택을 통해 상대적으로 효율적인 알고리즘 개발 가능
+ 구현에 따른 자료구조
  + List
  + Tuple
  + Linked list
  + Circular linked list
  + Doubly linked list
  + Hash table
+ 형태에 따른 자료구조
  + Linear
    + Stack
    + Queue
    + Deque
  + Non-Linear
    + Graph
    + Tree

<!-- More -->

***

# Queue

<img width="647" alt="Queue" src="https://user-images.githubusercontent.com/42334717/180357720-db9127a2-8b5d-4d73-aa71-67d7186b3478.png">

+ Method
  + `enQueue`: Insert
  + `deQueue`: Delete
+ Application
  + 프린터의 인쇄 대기열
  + 은행 업무
  + 프로세스 관리
+ Queue underflow: queue가 비어있을 때 `queue.deQueue`를 사용한 경우 발생
+ Queue overflow: queue가 가득차있을 때 `queue.enQueue`를 사용한 경우 발생

~~~python
class Queue():
    def __init__(self):
        self.queue = []

    def enQueue(self, n):
        self.queue.append(n)

    def deQueue(self):
        if len(self.queue) == 0:
            return -1
        return self.queue.pop(0)

    def printQueue(self):
        print(self.queue)
~~~

![Queue](https://user-images.githubusercontent.com/42334717/180360564-2004a21f-3c33-4ff5-a22a-ad98e0c4fecc.png)

***

# Stack

<img width="647" alt="Stack" src="https://user-images.githubusercontent.com/42334717/180357599-be2205fc-1569-46e8-a8de-5d3b3ddc7cda.png">

+ Method
  + `push`: Insert
  + `pop`: Delete
+ Application
  + Web browser 방문 기록
  + 실행 취소 (undo)
  + 후위 표기법 계산
+ Stack underflow: stack이 비어있을 때 `stack.pop`를 사용한 경우 발생
+ Stack overflow: stack이 가득차있을 때 `stack.push`를 사용한 경우 발생

~~~python
class Stack():
    def __init__(self):
        self.stack = []

    def push(self, n):
        self.stack.append(n)

    def pop(self):
        if len(self.stack) == 0:
            return -1
        return self.stack.pop()

    def printStack(self):
        print(self.stack)
~~~

![Stack](https://user-images.githubusercontent.com/42334717/180360631-edf403ba-21a0-48bf-8b15-3afd8d569e99.png)

***

# Queue vs. Stack

||Queue|Stack|
|:-:|:-:|:-:|
|순서 유무|O|O|
|삽입 (insert)|List의 rear|List의 top|
|삭제 (delete)|List의 front|List의 top|
|구조 (structure)|First-In-First-Out (FIFO)|Last-In-First-Out (LIFO)|