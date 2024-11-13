---
title: Python의 빠른 연산을 위한 Process 기반 병렬 처리
date: 2023-08-02 22:01:57
categories:
- Etc.
tags:
- Python
---
# Introduction

회사에서 약 17,000장 가량의 고화질 이미지 데이터를 처리해야하는 일이 생겼는데 총 처리 시간이 약 10시간 가량 소요됐다.
너무 오랜 시간이 소요되기 때문에 이를 빠르게 바꿔보고자 병렬 처리를 해보려고 했다.
먼저 생각난 키워드는 multithreading이여서 [`threading`](https://docs.python.org/ko/3/library/threading.html) 라이브러리에 대해 찾아보게 되었다.
하지만 python은 한 thread가 python 객체에 대한 접근을 제어하는 [mutex](https://namu.wiki/w/%EB%AE%A4%ED%85%8D%EC%8A%A4)인 [GIL](https://ssungkang.tistory.com/entry/python-GIL-Global-interpreter-Lock%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C) (Global Interpreter Lock)이 존재하여 [CPU 작업이 적고 I/O 작업이 많은 처리에서 효과를 볼 수 있다](https://monkey3199.github.io/develop/python/2018/12/04/python-pararrel.html). (~~더 이상은 너무 어려워요,,,~~)
Cython에서는 [이렇게](https://github.com/Zerohertz/PANPP/blob/d518c688de448f91c8fd6d194aa1cc3494fb6aa0/models/post_processing/boxgen/boxgen.pyx#L35C36-L35C36) `nogil=True`로 정의해 GIL를 해제하고 병렬 처리를 할 수 있다.
그렇다면 현재 문제인 대량의 고화질 이미지 데이터를 최대한 빠르게 처리하려면 어떻게 해야할까?
이 경우에는 process 기반의 병렬 처리를 지원하는 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) 라이브러리를 사용하면 된다.

<!-- More -->

---

# multiprocessing

성능 비교를 위한 간단한 `multiprocessing`의 예제는 다른 글에 많아 본 글에서는 바로 어떻게 이 문제를 해결했는지 설명하겠다.
먼저 task를 수행할 `main()` 함수에 4가지 변수가 사용되어 `multiprocessing.Pool.map()` 메서드를 사용하지 못하여 `multiprocessing.Pool.starmap()` 메서드를 사용했다.
`multiprocessing.Pool(processes=${NUM_POOL})`을 통해 process의 수를 정의할 수 있다. (`multiprocessing.cpu_count()` 메서드를 통해 현재 기기의 CPU 코어 수를 확인할 수 있다.)
마지막으로 process가 처리할 이미지를 정의하기 위해 `args`의 마지막 인덱스에 각 process가 처리할 이미지의 인덱스를 포함시켰다.

```python
import multiprocessing as mp
...
def main(org, tar, dv, ant):
    ...
def run():
    ...
    NUM_POOL = ${NUM_POOL}

    args = [[org, tar, dv, []] for _ in range(NUM_POOL)]
    for i in range(len(annotations)):
        args[i % NUM_POOL][3].append(i)
    with mp.Pool(processes=NUM_POOL) as pool:
        res = pool.starmap(main, args)
    ...
if __name__ == "__main__":
    run()
```

`multiprocessing`의 성능을 확인하기 위해 이미지 100장에 대해 실험을 진행했고 결과는 아래와 같다.

```yaml
ORG:            210.292초
processes=1:    201.165초
processes=25:   116.874초
processes=50:   65.589초
processes=64:   55.173초
processes=128:  35.111초
```

확실히 `multiprocessing`를 사용하지 않는 것 (`ORG`)에 비해 빠름을 확인할 수 있고, process의 수를 증가시킴에 따라 처리 시간이 짧아짐을 확인할 수 있다.
하지만 CPU 코어 수는 아래와 같이 64개인데 process의 수를 128로 설정해도 잘 실행됐고 오히려 빨랐다. (~~ChatGPT는 이러지 말라긴 하는데,,,~~)

![CPU](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/257830180-ea23c216-c877-4d92-bba1-7894e3140096.png)

~~불타는 CPU~~

혹은 아래와 같이 진행해도 같다. (~~굳이 `starmap` 메서드를 쓸 이유가 없었다...~~)

```python
import multiprocessing as mp
...
def main(args):
    ant, org, tar, dv = args
    ...
def run():
    ...
    NUM_POOL = ${NUM_POOL}

    args = [[ant, org, tar, dv] for ant in annotations]
    with mp.Pool(processes=NUM_POOL) as pool:
        res = pool.map(main, args)
    ...
if __name__ == "__main__":
    run()
```

---

# Progress Bar

단일 process에서 반복문을 실행할 때 `tqdm` 라이브러리를 사용하여 진행 상황을 확인할 수 있다.
하지만 병렬 처리 시 확인하기 쉽지 않아 아래와 같은 방법을 찾았다.

```python 1. multiprocessing
import multiprocessing as mp
with mp.Pool(processes=NUM_POOL) as pool:
    res = list(tqdm(pool.imap(main, args), total=len(args)))
```

```python 2. tqdm
from tqdm.contrib.concurrent import process_map
res = process_map(main, args, max_workers=NUM_POOL)
```

```python 3. parmap
import parmap
res = parmap.map(main, args, pm_pbar=True, pm_processes=NUM_POOL)
```

실행 시간 자체는 모두 동일한 것으로 확인했다.