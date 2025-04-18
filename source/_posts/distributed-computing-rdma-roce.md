---
title: Distributed Computing with RDMA and RoCE
date: 2025-04-18 11:10:49
categories:
  - 4. MLOps
tags:
  - Computer Science
---

# Introduction

LLM이 발전하면서 distributed training, serving이 점점 더 중요해지고 있다.
분산 환경에서 cluster 간 통신은 필수적이고, 저지연/고처리량 network가 핵심이다.
PyTorch, DeepSpeed, vLLM 등 다양한 framework로 쉽게 구현할 수 있지만 ([물론,,, 처음엔 삽질을...](https://github.com/vllm-project/vllm/discussions/11353)), 이러한 기술들의 이해도를 높히기 위해 본 글을 작성한다.

<!--More-->

# NCCL (NVIDIA Collective Communication Library)

[공식 페이지](https://developer.nvidia.com/nccl)에선 아래와 같이 NCCL을 소개하고 있다.

{% cq %}
The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking.
NCCL provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter as well as point-to-point send and receive that are optimized to achieve high bandwidth and low latency over PCIe and NVLink high-speed interconnects within a node and over NVIDIA Mellanox Network across nodes.
{% endcq %}

<img src="/images/distributed-computing-rdma-roce/nccl.png" alt="nccl" width="762" />

위에서 나온 단어들을 정리해보면 아래와 같다.

- all-gather: 모든 process가 각자의 data를 모아 하나의 큰 dataset을 만드는 통신 방식
- all-reduce: 모든 process가 각자의 data를 모아 연산을 수행한 후 결과를 모든 process에 전달하는 통신 방식
- broadcast: 한 process의 data를 모든 process에 전달하는 통신 방식
- reduce: 모든 process의 data를 모아 연산을 수행한 후 결과를 하나의 process에 전달하는 통신 방식
- reduce-scatter: reduce와 all-gather의 결합으로, data를 모아 연산을 수행한 후 결과를 분산시키는 통신 방식
- PCIe (peripheral component interconnect express): 고속 직렬 computer 확장 bus 표준으로, GPU, SSD 등 다양한 hardware와의 통신에 사용
- [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/): NVIDIA의 고속 interconnect 기술로, GPU 간의 data 전송 속도를 높이기 위해 사용
- NVIDIA Mellanox Network: NVIDIA가 인수한 Mellanox Technologies의 network solution으로, 고성능 computing 및 data center network 최적화

## Key Features

- AMD, ARM, PCI Gen4 및 IB HDR에서 고대역폭 경로에 대한 자동 topology 감지
- [SHARPV2](https://docs.nvidia.com/networking/display/sharpv261) (scalable hierarchical aggregation and reduction protocol)를 활용한 network 내 all reduce 작업으로 최대 2배의 peak 대역폭
- 최고 대역폭과 최저 지연 시간을 가진 optimal set of rings and trees를 graph 검색
- Multi-threaded 및 multi-process applications 지원
- InfiniBand verbs, libfabric, RoCE 및 IP Socket internode 통신
- InfiniBand Adaptive routing으로 traffic 경로 변경 및 혼잡한 port 완화

## InfiniBand

InfiniBand는 HPC 클러스터, data center에서 많이 쓰는 고속/저지연 network다.
대역폭, 신뢰성 모두 뛰어나고, RDMA를 native로 지원해서 CPU 개입 없이 node 간 memory 직접 전송이 가능하다.
이 덕분에 대규모 분산 학습, 병렬 계산에서 필수적인 기술로 자리잡았다.

## MPI (Message Passing Interface)

메시지 전달 interface (MPI)는 분산 computing 환경에서 process 간의 통신을 가능하게 하는 표준화된 protocol이다.
MPI는 다양한 통신 pattern을 지원하며, 확장 가능한 고성능 응용 program 개발의 초석 역할을 해왔다.
NCCL과 같은 library와의 통합을 통해 MPI는 수많은 computing node 간 복잡한 data 교환을 효율적으로 조율하며, 집합 연산이 원활하게 진행되도록 돕는다.
MPI와 NCCL의 시너지는 전통적인 통신 모델이 현대의 분산 AI 및 HPC 응용 프로그램의 요구를 충족하도록 진화하고 있음을 보여준다.

---

# RDMA

RDMA(Remote Direct Memory Access)는 network를 통해 원격 memory에 직접 접근할 수 있는 기술이다.
RDMA를 사용하면 CPU의 개입 없이 data 전송이 이루어지므로, 지연 시간이 줄어들고 data 전송 속도가 향상된다.
RDMA는 InfiniBand, RoCE 등의 network 기술에서 지원된다.

## RDMA의 실제 활용 예시

RDMA는 대규모 분산 deep learning 학습, HPC (High Performance Computing), database cluster, 분산 file system (ex. Ceph, Lustre) 등에서 널리 사용된다.
예를 들어, PyTorch의 `DistributedDataParallel`, Horovod, vLLM 등에서 RDMA를 지원하는 backend를 사용하면, GPU 간 통신 병목을 최소화할 수 있다.

## RDMA의 주요 장점

- 낮은 latency: CPU intervention 없이 memory-to-memory 전송
- 높은 throughput: 대용량 data 전송에 최적화
- 낮은 CPU 사용률: CPU는 연산에 집중, network 전송은 NIC가 담당

---

# RoCE (RDMA over Converged Ethernet)

RDMA over Converged Ethernet (RoCE)은 RDMA의 이점을 ethernet network로 확장시켜, 보다 보편적인 ethernet infra에서도 고성능 computing을 실현할 수 있도록 한다.
RoCE는 RDMA의 직접 memory access 기능을 ethernet 환경에서 활용할 수 있게 하여, 비용 효율적이면서도 널리 지원되는 network solution을 제공한다.
이러한 protocol은 유연성과 확장성, 고성능이 요구되는 현대 data center에서 점점 더 많이 채택되고 있다.

## RoCE 환경 구축 시 고려사항

RoCE를 실제로 도입할 때는 아래와 같은 network 설정이 중요하다.

- MTU (Maximum Transmission Unit): RoCE는 packet 손실에 민감하므로, network switch와 NIC의 MTU를 9000 (jumbo frame) 등으로 일치시켜야 한다.
- PFC (Priority Flow Control): Ethernet 환경에서 lossless network를 구현하기 위해 PFC 설정이 필수적이다.
- ECN/RED: 혼잡 제어를 위해 ECN (Explicit Congestion Notification)이나 RED (Random Early Detection) 등도 고려해야 한다.
- QoS (Quality of Service): RDMA traffic이 일반 traffic에 의해 방해받지 않도록 QoS 정책을 적용한다.

## RoCE v1

RoCE v1은 RDMA 기능을 ethernet network에 처음으로 도입한 version이다.
RDMA frame을 ethernet frame 안에 직접 capsule화하여, 특수한 network fabric 없이도 RDMA의 저지연 이점을 활용할 수 있도록 한다.
다만, RoCE v1은 일반적으로 단일 ethernet broadcast domain에 국한되기 때문에, 상대적으로 제한된 network 환경에서 최적의 성능을 발휘한다.
이러한 한계에도 불구하고, RoCE v1은 초저지연이 중요한 data center 환경에서 효과적인 solution으로 자리잡아 왔다.

## RoCE v2

RoCE v1의 강점을 기반으로 한 RoCE v2는 확장성과 상호 운용성을 크게 개선한 version이다.
RoCE v2는 RDMA frame을 UDP/IP packet에 capsule화하여, 더 큰 network topology와 다양한 network segment 간의 routing을 가능하게 한다.
이러한 확장은 network architecture 설계에 유연성을 제공할 뿐만 아니라, 복잡한 multi-tenant data center 환경에서 필수적인 고급 혼잡 제어 mechanism을 지원한다.

RoCE v2가 표준 IP network 상에서 routing을 지원함으로써, 물리적으로 분산된 resource 간의 분산 computing 작업에 특히 유리하다.
현대 network infra에 최적화된 RoCE v2는 전통적인 InfiniBand solution이 제약되거나 비용 효율적이지 않은 경우에 고성능 통신을 실현할 수 있는 대안으로 각광받고 있다.

| Feature          | RoCE v1                               | RoCE v2                                |
| ---------------- | ------------------------------------- | -------------------------------------- |
| Encapsulation    | Ethernet frame                        | UDP/IP packet                          |
| Routing          | L2 (Layer 2, same broadcast domain)   | L3 (Layer 3, supports IP routing)      |
| 사용 환경        | 단일 ethernet broadcast domain        | 대규모, multi-tenant, 라우팅 환경      |
| 확장성           | 제한적                                | 매우 우수                              |
| 혼잡 제어        | 제한적                                | 고급 혼잡 제어 mechanism 지원          |
| Interoperability | 낮음                                  | 높음                                   |
| Typical Use Case | Ultra-low-latency, 소규모 data center | 대규모 data center, 분산 AI/cloud 환경 |

## InfiniBand vs RoCE

| 항목         | InfiniBand                   | RoCE                                         |
| ------------ | ---------------------------- | -------------------------------------------- |
| Network type | 전용 InfiniBand fabric       | Ethernet 기반                                |
| 구축 비용    | 별도 장비 필요, 비용 높음    | 상대적으로 저렴, 기존 infra 활용 가능        |
| 성능         | 최고 성능, ultra-low latency | Lossless Ethernet 환경에서 InfiniBand에 근접 |
| 호환성       | HPC/AI 특화, 범용성 낮음     | 범용성 높음, 기존 infra와 통합 용이          |
| 관리 복잡도  | 전용 환경, 관리 일관성       | Ethernet tuning 필요, 설정 복잡할 수 있음    |

## 최신 동향 및 활용 사례

- Meta, Microsoft, Google 등 hyperscaler들은 대규모 AI 학습 클러스터에 RoCE 기반 network를 적극 도입 중
- 최근에는 RoCE v2와 AI/ML workload에 최적화된 switch NIC가 출시되고 있다.
- Open sources
  - [OpenFabrics Alliance](https://www.openfabrics.org/)
  - [rdma-core](https://github.com/linux-rdma/rdma-core)

---

{% note References %}

- [NVIDIA: NVIDIA Collective Communication Library (NCCL) Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NVIDIA: RoCEv2](https://docs.nvidia.com/networking/display/winofv55054000/rocev2)
- [FS: An In-Depth Guide to RoCE v2 Network](https://www.fs.com/blog/an-indepth-guide-to-roce-v2-network-2266.html)
- [Meta: RoCE networks for distributed AI training at scale](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/)

{% endnote %}
