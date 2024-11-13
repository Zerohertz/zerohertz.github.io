---
title: DFS and BFS
date: 2022-11-07 15:17:10
categories:
- 1. Algorithm
tags:
- Python
---
# Graph

> Graph: 연결되어있는 원소간의 관계를 표현한 자료구조

+ 정점 (Vertex): 연결할 객체
+ 간선 (Edge): 객체 연결
+ 그래프 `G` = (`V`, `E`)
  + `V`: 정점의 집합
  + `E`: 간선의 집합
+ 차수 (Degree): 정점에 부속되어 있는 간선의 수
  + 진입차수 (In-degree): 정점을 머리로 하는 간선의 수
  + 진출차수 (Out-degree): 정점을 꼬리로 하는 간선의 수
+ 경로 (Path): 정점 $V_i$에서 $V_j$까지 간선으로 연결된 정점을 순서대로 나열한 리스트
  + 단순 경로 (Simple path): 모두 다른 정점으로 구성된 경로
  + 경로 길이 (Path length): 경로를 구성하는 간선의 수
  + 사이클 (Cycle): 단순 경로 중에서 경로의 시작 정점과 마지막 정점이 같은 경로

~~~python
'''
N: 정점의 개수
M: 간선의 개수
V: 탐색을 시작할 정점
'''
import sys
read = sys.stdin.readline

N, M, V = map(int, read().split())
g = [[] for _ in range(N + 1)]

'''
정점 a와 b의 간선 연결 구현
'''

for _ in range(M):
  a, b = map(int, read().split())
  g[a].appned(b)
  g[b].append(a)
~~~

~~~python
4 5 1 # N, M, V
1 2 # a, b
1 3
1 4
2 4
3 4
> g
[[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]]
~~~

<!-- More -->

***

# Depth-First Search, DFS

> 깊이 우선 탐색: 최대한 깊이 내려가며 더 깊이 이동이 불가할 때 옆으로 이동

~~~python
v = [False for _ in range(N + 1)]

def DFS(initPos):
  print(initPos, end = ' ')
  v[initPos] = True
  for i in g[initPos]:
    if not v[i]:
      DFS(i)
      v[i] = True

DFS(V)
~~~

1. 현재 위치 방문 `True`
2. 그래프 내에서 현재 위치와 연결된 정점들 (`g[initPos]`)에 따라 방문하지 않았다면 재귀적으로 1번 실행

***

# Breadth-First Search, BFS

> 너비 우선 탐색: 최대한 내려가지 않으며 같은 깊이에서 이동이 불가할 때 아래로 이동

~~~python
from collections import deque

q = deque()
v = [False for _ in range(N + 1)]

def BFS(initPos):
  q.append(initPos)
  v[initPos] = True
  while q:
    tmp = q.popleft()
    print(tmp, end = ' ')
    for i in g[tmp]:
      if not v[i]:
        v[i] = True
        q.append(i)

BFS(V)
~~~

1. `deque`에 현재 위치 `append()` 및 현재 위치 방문 `True`
2. `tmp` 변수에 `deque`의 최신 값을 `pop`하여 현재 위치 업데이트
3. 그래프 내에서 현재 위치와 연결된 정점들 (`g[tmp]`)에 따라 방문하지 않았다면 `deque`에 `append()`

***

# BOJ: 1260

> 문제

그래프를 DFS로 탐색한 결과와 BFS로 탐색한 결과를 출력하는 프로그램을 작성하시오. 단, 방문할 수 있는 정점이 여러 개인 경우에는 정점 번호가 작은 것을 먼저 방문하고, 더 이상 방문할 수 있는 점이 없는 경우 종료한다. 정점 번호는 1번부터 N번까지이다.

> 입력

첫째 줄에 정점의 개수 N(1 ≤ N ≤ 1,000), 간선의 개수 M(1 ≤ M ≤ 10,000), 탐색을 시작할 정점의 번호 V가 주어진다. 다음 M개의 줄에는 간선이 연결하는 두 정점의 번호가 주어진다. 어떤 두 정점 사이에 여러 개의 간선이 있을 수 있다. 입력으로 주어지는 간선은 양방향이다.

> 출력

첫째 줄에 DFS를 수행한 결과를, 그 다음 줄에는 BFS를 수행한 결과를 출력한다. V부터 방문된 점을 순서대로 출력하면 된다.

~~~python
import sys
from collections import deque
read = sys.stdin.readline

def DFS(initPos):
  print(initPos, end = ' ')
  v[initPos] = True
  for i in g[initPos]:
    if not v[i]:
      DFS(i)
      v[i] = True

def BFS(initPos):
  q.append(initPos)
  v[initPos] = True
  while q:
    tmp = q.popleft()
    print(tmp, end = ' ')
    for i in g[tmp]:
      if not v[i]:
        v[i] = True
        q.append(i)

N, M, V = map(int, read().split())
g = [[] for _ in range(N + 1)]

for _ in range(M):
  a, b = map(int, read().split())
  g[a].append(b)
  g[b].append(a)

for i in range(N + 1):
  g[i].sort()

v = [False for _ in range(N + 1)]
DFS(V)
print()

q = deque()
v = [False for _ in range(N + 1)]
BFS(V)
~~~