---
title: Kubernetes 입문기
date: 2023-06-18 15:11:10
categories:
- 3. DevOps
tags:
- Kubernetes
---
# Introduction

+ Kubernetes$\_{^{\ [1]}}$: Container화된 application을 자동화하고 관리하기 위한 open source container orchestration 플랫폼$\_{^{\ [2,3]}}$
  + MLOps에서 왜 Kubernetes가 필요할까?$\_{^{\ [4]}}$
    + 수많은 머신러닝 모델의 학습 요청을 차례대로 실행하는 것
    + 다른 작업 공간에서도 같은 실행 환경을 보장해야 하는 것
    + 배포된 서비스에 장애가 생겼을 때 빠르게 대응해야 하는 것
    + Etc.
  + Cluster: Container화된 application을 실행하는 node라고 하는 worker machine의 집합
    + 모든 cluster는 최소 한 개의 worker node를 가짐
  + Kubectl: Kubernetes의 cluster와 통신하여 다양한 object들의 상태 확인 또는 CRUD 작업 등을 위해 사용되는 CLI 도구$\_{^{\ [5]}}$
    + CRUD: Create (생성), Read (읽기), Update (갱신), Delete (삭제)
  + Node: Cluster 내에서 workload를 실행할 수 있는 하드웨어 또는 가상 머신 인스턴스$\_{^{\ [6]}}$
    + Cluster에 따라 가상 또는 물리적 머신일 수 있음
    + Control plane에 의해 관리되며 pod를 실행하는 데 필요한 서비스를 포함
  + Pod: Cluster의 가장 작은 배포 단위
    + 한 개 이상의 container로 구성
    + Storage 및 network를 공유하고 함께 batch 및 scheduling (공유 context)
  + etcd: Cluster의 데이터를 저장하는 key-value 기반 DB
  + Namespace: Cluster 내의 논리적 분리 단위
    + Resource를 논리적으로 나누기 위한 방법
    + 사용자별 접근 권한 부여 가능
+ Minikube$\_{^{\ [7]}}$: 로컬 개발 환경에서 단일 node Kubernetes cluster를 실행하기 위한 도구

<!-- More -->

---

# Minikube, K3s, K8s?

|Feature|Minikube|K3s|K8s (기본 Kubernetes)|
|:-:|:-:|:-:|:-:|
|설치의 용이성|로컬 개발에 최적화, 쉽게 설치 가능|단일 바이너리로 빠르고 간단한 설치|복잡한 설정과 배포 요구|
|메모리 사용량|일반적으로 더 많은 메모리 사용|경량화, 적은 메모리 사용|메모리 사용량은 설정에 따라 다름|
|운영 체제 지원|대부분의 주요 OS 지원|대부분의 Linux 배포판 지원|광범위한 OS 및 플랫폼 지원|
|클러스터 구성|주로 단일 노드 클러스터|멀티 노드 클러스터 지원 가능|다양한 멀티 노드 클러스터 구성 가능|
|확장성|주로 개발 및 테스팅 목적|소규모에서 중규모 규모까지 확장 가능|대규모 클러스터를 위한 확장성|
|추가 기능|로컬 개발에 특화된 추가 도구 및 기능 제공|IoT, 엣지 컴퓨팅 등에 특화|광범위한 플러그인 및 확장 지원|
|관리 복잡성|적음|중간|높음|

---

# Minikube

Minikube는 아래와 같이 시작할 수 있다.

```shell Cluster 시작
$ minikube start --kubernetes-version=v1.26.3
😄  Darwin 13.4 (arm64) 의 minikube v1.30.1
✨  자동적으로 docker 드라이버가 선택되었습니다
📌  Using Docker Desktop driver with root privileges
👍  minikube 클러스터의 minikube 컨트롤 플레인 노드를 시작하는 중
🚜  베이스 이미지를 다운받는 중 ...
💾  쿠버네티스 v1.26.3 을 다운로드 중 ...
    > preloaded-images-k8s-v18-v1...:  330.52 MiB / 330.52 MiB  100.00% 26.59 M
    > gcr.io/k8s-minikube/kicbase...:  336.39 MiB / 336.39 MiB  100.00% 8.08 Mi
🔥  Creating docker container (CPUs=2, Memory=4000MB) ...
🐳  쿠버네티스 v1.26.3 을 Docker 23.0.2 런타임으로 설치하는 중
    ▪ 인증서 및 키를 생성하는 중 ...
    ▪ 컨트롤 플레인이 부팅...
    ▪ RBAC 규칙을 구성하는 중 ...
🔗  Configuring bridge CNI (Container Networking Interface) ...
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v5
🔎  Kubernetes 구성 요소를 확인...
🌟  애드온 활성화 : storage-provisioner, default-storageclass
🏄  끝났습니다! kubectl이 "minikube" 클러스터와 "default" 네임스페이스를 기본적으로 사용하도록 구성되었습니다.
$ docker ps
CONTAINER ID   IMAGE                                 COMMAND                  CREATED          STATUS          PORTS                                                                                                                                  NAMES
0051d46b67a8   gcr.io/k8s-minikube/kicbase:v0.0.39   "/usr/local/bin/entr…"   38 seconds ago   Up 37 seconds   127.0.0.1:51959->22/tcp, 127.0.0.1:51955->2376/tcp, 127.0.0.1:51957->5000/tcp, 127.0.0.1:51958->8443/tcp, 127.0.0.1:51956->32443/tcp   minikube
```

```shell Cluster 일시정지
$ minikube pause
⏸️  Pausing node minikube ...
⏯️  Paused 18 containers in: kube-system, kubernetes-dashboard, storage-gluster, istio-operator
```

```shell Cluster 재가동
$ minikube unpause
⏸️  Unpausing node minikube ...
⏸️  Unpaused 18 containers in: kube-system, kubernetes-dashboard, storage-gluster, istio-operator
```

```shell Cluster 종료
$ minikube stop
✋  "minikube" 노드를 중지하는 중 ...
🛑  "minikube"를 SSH로 전원을 끕니다 ...
🛑  1개의 노드가 중지되었습니다.
```

Ubuntu 환경에서 부팅 시 바로 Minikube를 시작하려면 아래 단계를 거치면 된다.

```shell
$ minikube config set cpus ${NUM_CPU}
$ minikube config set memory ${MEMORY}
$ minikube config set driver ${DRIVER}
$ sudo vi /etc/systemd/system/minikube.service
[Unit]
Description=Minikube
After=network.target

[Service]
Type=oneshot
User=${USER_NAME}
RemainAfterExit=yes
ExecStart=/usr/local/bin/minikube start
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
$ sudo systemctl daemon-reload
$ sudo systemctl enable minikube.service
```

자 그럼 Pod CRUD (**C**reate, **R**ead, **U**pdate, **D**elete)에 대해서 알아보자!

## Create

> Imperative (명령형)
> ```shell
$ kubectl run k8s-is-so-hard-name --image=nginx
pod/k8s-is-so-hard-name created
```

> Declarative (선언형)
> ```yaml pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: k8s-is-so-hard-name2
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
    ports:
    - containerPort: 80
```
> ```shell
$ kubectl apply -f pod.yaml
pod/k8s-is-so-hard-name2 created
```

<details>
<summary>
Kubernetes는 _를 싫어해요,,,
</summary>

The Pod "k8s_is_so_hard_name" is invalid: metadata.name: Invalid value: "k8s_is_so_hard_name": a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')

</details>

## Read

```shell
$ kubectl describe pods k8s-is-so-hard-name2
Name:             k8s-is-so-hard-name2
Namespace:        default
Priority:         0
Service Account:  default
Node:             minikube/192.168.49.2
Start Time:       Mon, 26 Jun 2023 19:55:03 +0900
Labels:           <none>
Annotations:      <none>
Status:           Running
IP:               10.244.0.13
IPs:
  IP:  10.244.0.13
Containers:
  nginx:
    Container ID:   docker://3d48d61d20790d8df10bee2876b302e8d24cc2867e3d05682f100ccdc03e6e0c
    Image:          nginx:1.14.2
    Image ID:       docker-pullable://nginx@sha256:f7988fb6c02e0ce69257d9bd9cf37ae20a60f1df7563c3a2a6abe24160306b8d
    Port:           80/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Mon, 26 Jun 2023 19:55:03 +0900
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-b59qj (ro)
Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes:
  kube-api-access-b59qj:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   BestEffort
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  3m2s  default-scheduler  Successfully assigned default/k8s-is-so-hard-name2 to minikube
  Normal  Pulled     3m2s  kubelet            Container image "nginx:1.14.2" already present on machine
  Normal  Created    3m2s  kubelet            Created container nginx
  Normal  Started    3m2s  kubelet            Started container nginx
```

## Update

```shell
$ kubectl edit pods k8s-is-so-hard-name2
```

<img width="1082" alt="Update" src="/images/k8s/2b13f1ed-ed99-49c5-9b49-152ec76e2b2a">

vi editor를 통한 실행 중인 pod 수정 가능

## Delete

> Imperative (명령형)
> ```shell
$ kubectl delete pods k8s-is-so-hard-name
pod "k8s-is-so-hard-name" deleted
```

> Declarative (선언형)
> ```shell
$ kubectl delete -f pod.yaml
pod "k8s-is-so-hard-name2" deleted
```

---

# K3s

[K3s](https://k3s.io/)는 매우 쉽게 설치하고 삭제할 수 있다.

```bash install_k3s.sh
curl -sfL https://get.k3s.io | sh -s - --docker
sudo cat /etc/rancher/k3s/k3s.yaml
rm -rf ~/.kube
mkdir ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
```

```bash remove_k3s.sh
/usr/local/bin/k3s-uninstall.sh
```

실행, 상태 확인, 종료는 아래와 같다.

```shell
$ sudo systemctl start k3s
$ sudo systemctl status k3s
● k3s.service - Lightweight Kubernetes
     Loaded: loaded (/etc/systemd/system/k3s.service; enabled; vendor preset: enabled)
     Active: active (running) since Mon 2023-08-07 23:27:59 KST; 24h ago
       Docs: https://k3s.io
    Process: 588152 ExecStartPre=/bin/sh -xc ! /usr/bin/systemctl is-enabled --quiet nm-cloud-setup.service (code=exited, status=0/SUCCESS)
    Process: 588154 ExecStartPre=/sbin/modprobe br_netfilter (code=exited, status=0/SUCCESS)
    Process: 588155 ExecStartPre=/sbin/modprobe overlay (code=exited, status=0/SUCCESS)
   Main PID: 588156 (k3s-server)
      Tasks: 26
     Memory: 649.0M
        CPU: 1h 46min 41.000s
     CGroup: /system.slice/k3s.service
             └─588156 "/usr/local/bin/k3s server" "" "" "" "" "" "" "" "" ""

Aug 08 23:47:57 0hz k3s[588156]: I0808 23:47:57.051554  588156 handler.go:232] Adding GroupVersion helm.cattle.io v1 to ResourceManager
Aug 08 23:47:57 0hz k3s[588156]: I0808 23:47:57.051643  588156 handler.go:232] Adding GroupVersion k3s.cattle.io v1 to ResourceManager
Aug 08 23:47:57 0hz k3s[588156]: I0808 23:47:57.051729  588156 handler.go:232] Adding GroupVersion traefik.containo.us v1alpha1 to ResourceManager
Aug 08 23:47:57 0hz k3s[588156]: I0808 23:47:57.051841  588156 handler.go:232] Adding GroupVersion helm.cattle.io v1 to ResourceManager
$ sudo systemctl stop k3s
```

---

# K8s

```bash
# Docker
$ sudo apt-get update
$ sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
$ sudo apt-get update
$ sudo apt-get install -y docker-ce docker-ce-cli containerd.io
$ sudo systemctl enable docker
$ sudo systemctl start docker
$ sudo systemctl enable containerd
$ sudo systemctl start containerd
$ sudo usermod -aG docker ${USER}
$ sudo systemctl reboot

# K8s
$ sudo swapoff -a && sudo sed -i '/swap/s/^/#/' /etc/fstab
$ cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
br_netfilter
EOF
$ cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF
$ sudo sysctl --system
$ sudo apt-get update
$ sudo apt-get install -y apt-transport-https ca-certificates curl
$ curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
$ sudo apt-get update
$ sudo apt-get install -y kubelet kubeadm kubectl
$ sudo apt-mark hold kubelet kubeadm kubectl
$ sudo rm /etc/containerd/config.toml
$ systemctl restart containerd
$ sudo kubeadm init
$ mkdir -p $HOME/.kube
$ sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
$ sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

> CNI (Container Network Interface): Kubernetes에서 pod 간의 networking 처리

```shell
$ kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
$ kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$(kubectl version | base64 | tr -d '\n')"
$ kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

|Name|Pros|Cons|
|:-:|:-:|:-:|
|Calico|- 정책 기반의 보안<br />- 확장성이 뛰어남<br />- 네트워크 정책 지원|- 설정과 운영이 복잡할 수 있음<br />- 다양한 네트워크 토폴로지와 상호 작용할 때|
|Weave Net|- 설치 및 설정이 간단<br />- 다양한 환경과 토폴로지에서 동작<br />- 클라우드, 온-프레미스, 하이브리드 환경 모두 지원|- 대규모 클러스터에서 성능 저하 가능<br />- 직접 연결된 호스트간 통신에 제한이 있을 수 있음|
|Flannel|- 설치 및 설정이 간단<br />- 다양한 백엔드 옵션 (UDP, VXLAN 등)<br />- 작은 규모의 클러스터에서 잘 작동|- 고급 네트워킹 기능 제한<br />- 네트워크 정책 지원이 제한적<br />- 확장성과 성능이 다른 옵션에 비해 떨어질 수 있음|

아래 명령어를 통해 worker node를 등록할 수 있다.

```shell
$ kubeadm join ${MASTER_NODE_IP}:6443 --token ${TOKEN} --discovery-token-ca-cert-hash sha256:${HASH}
```

단일 node에서 사용할 경우 아래 과정을 거치면 된다.

```shell
$ kubectl describe node ${MASTER_NODE_NAME} | grep Taints
$ kubectl taint nodes ${MASTER_NODE_NAME} node-role.kubernetes.io/control-plane:NoSchedule-
```

마지막으로 삭제는 아래 명령어로 진행할 수 있다.

```shell
$ kubectl drain ${NODE_NAME} --delete-local-data --force --ignore-daemonsets
$ kubectl delete node ${NODE_NAME}
$ sudo kubeadm reset
$ sudo apt-get purge kubeadm kubectl kubelet kubernetes-cni kube*
$ sudo apt-get autoremove
$ rm -rf ~/.kube
```

최신 버전을 위처럼 생으로 사용하면 어려운 점이 많을 수 있으니 안정적인 버전을 채택하여 아래와 같이 사용토록 하자...

```bash
sudo swapoff -a && sudo sed -i '/swap/s/^/#/' /etc/fstab
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
br_netfilter
EOF
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF
sudo sysctl --system
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
sudo apt-get update
# sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-get install -y kubelet=1.22.8-00 kubeadm=1.22.8-00 kubectl=1.22.8-00
sudo apt-mark hold kubelet kubeadm kubectl
# sudo rm /etc/containerd/config.toml
# systemctl restart containerd
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF
sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker

sudo kubeadm init --pod-network-cidr=10.244.0.0/16
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
# kubectl taint nodes --all node-role.kubernetes.io/control-plane:NoSchedule-
kubectl taint node 0hz node-role.kubernetes.io/master-
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

`kubectl`을 ZSH에서 자동완성하려면 아래의 코드를 추가한다.

```bash ~/.zshrc
source <(kubeadm completion zsh)
source <(kubectl completion zsh)
```

---

# References

1. [GitHub: kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)
2. [Kubernetes: 쿠버네티스란 무엇인가?](https://kubernetes.io/ko/docs/concepts/overview/)
3. [Red Hat: 쿠버네티스 아키텍처 소개](https://www.redhat.com/ko/topics/containers/kubernetes-architecture)
4. [모두의 MLOps: Why Kubernetes?/MLOps & Kubernetes](https://mlops-for-all.github.io/docs/introduction/why_kubernetes/#mlops--kubernetes)
5. [Kubernetes: kubectl](https://kubernetes.io/ko/docs/tasks/tools/#kubectl)
6. [Kubernetes: 노드](https://kubernetes.io/ko/docs/concepts/architecture/nodes/)
7. [GitHub: kubernetes/minikube](https://github.com/kubernetes/minikube)