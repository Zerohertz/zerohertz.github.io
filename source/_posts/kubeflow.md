---
title: Kubeflow
date: 2023-08-06 01:55:21
categories:
- 4. MLOps
tags:
- Kubernetes
- Kubeflow
mathjax: true
---
# Introduction

> Kubeflow: Machine learning 및 Deep learning workload를 위한 오픈 소스 플랫폼으로, Kubernetes 위에서 실행되며 ML workload의 배포, 관리 및 확장을 용이하게 해줍니다. Kubeflow는 Kubernetes의 높은 가용성, 확장성 및 관리성과 machine learning 작업에 필요한 도구와 라이브러리를 통합하여 종합적인 ML 작업 플로우를 제공합니다.

![Overview](/images/kubeflow/258590136-e51ffab9-cf90-41e7-b01e-f4fd38a33266.svg)

## Why Kubeflow?

1. Scalable ML Workload Management
   + Kubernetes의 강력한 scaling 및 관리 기능을 활용하여 ML workload를 효율적으로 배포하고 관리
   + 자동화된 scaling과 resource 관리는 복잡한 ML 작업에서 유용
2. Portable Environments
   + 인프라 구성을 추상화하고 portable한 ML 작업 환경을 제공하여 로컬 머신에서 클라우드까지 다양한 환경에서 일관된 방식으로 작업할 수 있는 환경 구축
3. Integrated Components
   + Machine learning 작업을 위한 다양한 컴포넌트를 통합하여 제공
   + Jupyter Notebook, TensorBoard, TensorFlow Serving, ...
4. Designed Pipelines
   + Kubeflow Pipelines를 통해 End-to-End machine learning workflow를 시각적으로 디자인 및 실행 가능
   + 반복적인 작업을 자동화하고 재현 가능한 실험 수행

<!-- More -->

## Pros & Cons

+ Pros
  + Scaling and Management: Kubernetes 플랫폼의 강력한 scaling 및 관리 기능을 활용하여 ML workload를 효율적으로 관리
  + Diverse Tools and Libraries: Kubeflow는 Jupyter 노트북, TensorBoard, Katib 등 다양한 툴과 machine learning 라이브러리를 통합하여 제공하여 개발자와 데이터 과학자의 생산성 향상
  + Portable Workspace: 인프라 추상화로 인해 로컬 머신부터 클라우드까지 일관된 환경에서 작업
  + End-to-End Workflows: Kubeflow Pipelines를 사용하여 데이터 전처리부터 모델 훈련, 배포까지의 workflow 통합적 관리
+ Cons
  + Learning Curve: Kubeflow의 복잡한 기능과 컴포넌트들을 익히는 데 시간 필요
  + Resource Overhead: Kubernetes 및 Kubeflow를 위한 resource가 높을 수 있으며, 작은 규모의 프로젝트에는 overhead가 발생 가능
  + Initial Setup: Kubeflow를 설정하고 실행하는 초기 단계가 상대적으로 복잡

---

# Features

![Kubeflow](/images/kubeflow/258588922-9a8cb79b-12ea-4e41-80ec-2297d79e03f2.png)

+ [Central Dashboard](https://www.kubeflow.org/docs/components/central-dash/): Web browser를 통해 dashboard UI로 Notebooks, Experiments (AutoML), Experiments (KFP) 등의 컴포넌트 이용 가능
+ [Kubeflow Notebooks](https://www.kubeflow.org/docs/components/notebooks/): Web browser에서 python 코드를 개발하고 실행할 수 있는 Jupyter Notebook 개발 도구 제공
+ [Training Operators](https://www.kubeflow.org/docs/components/training/): Tensorflow, PyTorch, MXNet 등 다양한 deep learning framework에 대한 분산 학습 지원
  + Machine learning 모델 코드를 담고 있는 docker image 경로와 분산 학습 cluster 정보 등을 정의한 명세서를 통해 workload 실행
+ [Experiments (AutoML)](https://www.kubeflow.org/docs/components/katib/): Machine learning 모델의 예측 정확도와 성능을 높이기 위한 반복 실험을 자동화하는 도구 
  + Katib를 사용하여 hyper parameter tuning 및 neural architecture search
  + Experiment (AutoML) 명세서 작성 후 Kubernetes에 배포 시 최적 모델 탐색
+ [KServe](https://www.kubeflow.org/docs/external-add-ons/kserve/): Kubernetes에 machine learning 모델을 배포하고 추론하는 기능 제공
  + Endpoint, transformer, predictor, explainer로 구성
  + Endpoint: Predictor에 데이터 전달
  + Transformer: 데이터 전처리, 후처리
  + Predictor: 전달된 데이터에 대해 추론
  + Explainer: 추론 결과에 대한 이유 제공 (XAI, eXplainable Artificial Intellgence)
  + Transformer와 explainer는 predictor와 연결하여 사용
+ [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/components/pipelines/): Machine learning workflow를 구축하고 배포하기 위한 ML workflow orchestration 도구
  + Pipeline components를 재사용하여 다양한 실험을 빠르고 쉽게 수행
  + DAG 형태로 workflow 구축
  + Workflow engine: Argo Workflow

---

# Installation

## K3s

```shell
$ sudo apt-get update
$ sudo apt-get install -y socat

# Docker
$ sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg lsb-release
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
$ echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
$ sudo apt-get update && apt-cache madison docker-ce
$ apt-cache madison docker-ce # 설치 가능한 버전 확인
$ sudo apt-get install -y containerd.io docker-ce=5:20.10.13~3-0~ubuntu-jammy docker-ce-cli=5:20.10.13~3-0~ubuntu-jammy
$ sudo docker run hello-world
Hello from Docker!
...
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
$ newgrp docker

# Swap Memory
$ sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
$ sudo swapoff -a

# Kubectl
$ curl -LO https://dl.k8s.io/release/v1.21.7/bin/linux/amd64/kubectl
$ sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
$ kubectl version --client
Client Version: version.Info{Major:"1", Minor:"21", GitVersion:"v1.21.7", GitCommit:"1f86634ff08f37e54e8bfcd86bc90b61c98f84d4", GitTreeState:"clean", BuildDate:"2021-11-17T14:41:19Z", GoVersion:"go1.16.10", Compiler:"gc", Platform:"linux/amd64"}

# K3s Cluster
$ curl -sfL https://get.k3s.io | sh -s - --docker
$ sudo cat /etc/rancher/k3s/k3s.yaml
...
$ mkdir .kube
$ sudo cp /etc/rancher/k3s/k3s.yaml .kube/config
$ sudo chown $USER:$USER .kube/config

# K3s Client
$ wget https://get.helm.sh/helm-v3.7.1-linux-amd64.tar.gz
$ tar -zxvf helm-v3.7.1-linux-amd64.tar.gz
$ sudo mv linux-amd64/helm /usr/local/bin/helm
$ wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv3.10.0/kustomize_v3.10.0_linux_amd64.tar.gz
$ tar -zxvf kustomize_v3.10.0_linux_amd64.tar.gz
$ sudo mv kustomize /usr/local/bin/kustomize
$ kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.20/deploy/local-path-storage.yaml
$ kubectl -n local-path-storage get pod
...
$ kubectl patch storageclass local-path  -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
$ kubectl get sc
...

# Start K3s
$ sudo systemctl start k3s
$ kubectl get nodes -o wide
NAME   STATUS   ROLES                  AGE     VERSION        INTERNAL-IP       EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION      CONTAINER-RUNTIME
0hz    Ready    control-plane,master   3m35s   v1.27.4+k3s1   192.168.219.200   <none>        Ubuntu 22.04.3 LTS   5.15.0-78-generic   containerd://1.7.1-k3s1
```

```yaml ~/.kube/config
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: 
    ...
    server: https://127.0.0.1:6443
  name: default
contexts:
- context:
    cluster: default
    user: default
  name: default
current-context: default
kind: Config
preferences: {}
users:
- name: default
  user:
    client-certificate-data:
    ...
    client-key-data:
    ...
```

## Setup

```bash
# Kubeflow
$ git clone -b v1.7.0 https://github.com/kubeflow/manifests.git
$ cd manifests
$ kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -
$ kubectl get po -n cert-manager
$ kustomize build common/cert-manager/kubeflow-issuer/base | kubectl apply -f -
$ kustomize build common/istio-1-16/istio-crds/base | kubectl apply -f -
$ kustomize build common/istio-1-16/istio-namespace/base | kubectl apply -f -
$ kustomize build common/istio-1-16/istio-install/base | kubectl apply -f -
$ kubectl get po -n istio-system
$ kustomize build common/dex/overlays/istio | kubectl apply -f -
$ kubectl get po -n auth
$ kustomize build common/oidc-authservice/base | kubectl apply -f -
$ kubectl get po -n istio-system -w
$ kustomize build common/kubeflow-namespace/base | kubectl apply -f -
$ kubectl get ns kubeflow
$ kustomize build common/kubeflow-roles/base | kubectl apply -f -
$ kubectl get clusterrole | grep kubeflow
$ kustomize build common/istio-1-16/kubeflow-istio-resources/base | kubectl apply -f -
$ kubectl get clusterrole | grep kubeflow-istio
$ kubectl get gateway -n kubeflow
$ kustomize build apps/pipeline/upstream/env/platform-agnostic-multi-user | kubectl apply -f -
$ kubectl get po -n kubeflow
$ kustomize build apps/katib/upstream/installs/katib-with-kubeflow | kubectl apply -f -
$ kubectl get po -n kubeflow | grep katib
$ kustomize build apps/centraldashboard/upstream/overlays/istio | kubectl apply -f -
$ kubectl get po -n kubeflow | grep centraldashboard
$ kustomize build apps/admission-webhook/upstream/overlays/cert-manager | kubectl apply -f -
$ kubectl get po -n kubeflow | grep admission-webhook
$ kustomize build apps/jupyter/notebook-controller/upstream/overlays/kubeflow | kubectl apply -f -
$ kubectl get po -n kubeflow | grep notebook-controller
$ kustomize build apps/jupyter/jupyter-web-app/upstream/overlays/istio | kubectl apply -f -
$ kubectl get po -n kubeflow | grep jupyter-web-app
$ kustomize build apps/profiles/upstream/overlays/kubeflow | kubectl apply -f -
$ kubectl get po -n kubeflow | grep profiles-deployment
$ kustomize build apps/volumes-web-app/upstream/overlays/istio | kubectl apply -f -
$ kubectl get po -n kubeflow | grep volumes-web-app
$ kustomize build apps/tensorboard/tensorboards-web-app/upstream/overlays/istio | kubectl apply -f -
$ kubectl get po -n kubeflow | grep tensorboards-web-app
$ kustomize build apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow | kubectl apply -f -
$ kubectl get po -n kubeflow | grep tensorboard-controller
$ kustomize build apps/training-operator/upstream/overlays/kubeflow | kubectl apply -f -
$ kubectl get po -n kubeflow | grep training-operator
$ kustomize build common/user-namespace/base | kubectl apply -f -
$ kubectl get profile

# Forwarding
$ kubectl port-forward --address 0.0.0.0 svc/ml-pipeline-ui -n kubeflow 8888:80             # http://${IP}:8888/#/pipelines
$ kubectl port-forward --address 0.0.0.0 svc/katib-ui -n kubeflow 8081:80                   # http://${IP}:8081/katib/
$ kubectl port-forward --address 0.0.0.0 svc/centraldashboard -n kubeflow 8082:80           # http://${IP}:8082/
$ kubectl port-forward --address 0.0.0.0 svc/istio-ingressgateway -n istio-system 8080:80   # http://${IP}:8080/dex/auth/local/login?back=&state=rxl67iq5lvggjjh2675lxdraq
```

```shell
$ kubectl get pods --all-namespaces
NAMESPACE                   NAME                                                    READY   STATUS    RESTARTS       AGE
kubeflow                    kubeflow-pipelines-profile-controller-6476b6cb9-wdmzm   1/1     Running   0              114s
kubeflow                    katib-db-manager-945f44ff7-xzplc                        1/1     Running   0              114s
auth                        dex-654995d5d9-zmdzd                                    1/1     Running   0              114s
kube-system                 local-path-provisioner-957fdf8bc-44vx6                  1/1     Running   0              114s
kubeflow                    admission-webhook-deployment-789dc56fbf-h5ksd           1/1     Running   0              114s
kube-system                 svclb-traefik-0a9db00b-2fjz5                            2/2     Running   0              113s
cert-manager                cert-manager-6bbf5c9c95-bq64d                           1/1     Running   0              114s
kubeflow                    metadata-envoy-deployment-b48db5966-t7rgg               1/1     Running   0              114s
cert-manager                cert-manager-cainjector-654b7d6546-rmwwb                1/1     Running   0              114s
istio-system                istiod-76d8b99b46-czvx4                                 1/1     Running   0              113s
kube-system                 traefik-64f55bb67d-672tb                                1/1     Running   0              112s
kubeflow                    metacontroller-0                                        1/1     Running   0              110s
cert-manager                cert-manager-webhook-5bc4685cd4-f5jnl                   1/1     Running   0              113s
kube-system                 coredns-77ccd57875-dghdq                                1/1     Running   0              114s
kubeflow                    volumes-web-app-deployment-588d46bb75-kzkww             2/2     Running   0              108s
kubeflow                    ml-pipeline-scheduledworkflow-578475988-fqx2q           2/2     Running   0              108s
kubeflow-user-example-com   ml-pipeline-ui-artifact-76476b5cfd-5p6st                2/2     Running   0              108s
kubeflow                    mysql-7d8b8ff4f4-wgjrr                                  2/2     Running   0              108s
kubeflow-user-example-com   ml-pipeline-visualizationserver-677c86b748-4vjs2        2/2     Running   0              108s
kubeflow                    katib-ui-bcc6df7cf-p6nf8                                2/2     Running   1 (107s ago)   109s
istio-system                istio-ingressgateway-c7dddd948-w645z                    1/1     Running   0              112s
kubeflow                    ml-pipeline-viewer-crd-6857ccc85c-j4htg                 2/2     Running   1 (106s ago)   109s
kubeflow                    tensorboards-web-app-deployment-d4c644d74-4mcgc         2/2     Running   0              108s
kubeflow                    jupyter-web-app-deployment-6b9c8d94ff-2bv2v             2/2     Running   0              107s
kubeflow                    metadata-writer-6f95b9588c-4djq4                        2/2     Running   0              108s
kubeflow                    cache-server-7bb7f46866-tq6zx                           2/2     Running   0              107s
kubeflow                    workflow-controller-d445fd59d-xcfsm                     2/2     Running   1 (104s ago)   107s
kubeflow                    centraldashboard-7df67fb75-lc7rt                        2/2     Running   0              107s
kubeflow                    ml-pipeline-persistenceagent-55bd585845-qcvkg           2/2     Running   0              107s
kubeflow                    minio-55464b6ddb-wt2tf                                  2/2     Running   0              107s
kubeflow                    cache-deployer-deployment-5b544bdc4d-kjf9d              2/2     Running   1 (103s ago)   107s
istio-system                authservice-0                                           1/1     Running   0              110s
kubeflow                    ml-pipeline-ui-7658c799f5-8gpvn                         2/2     Running   0              107s
kubeflow                    ml-pipeline-visualizationserver-7b45b7fd56-dgq5p        2/2     Running   0              107s
kubeflow                    notebook-controller-deployment-c57f69568-mx7vs          2/2     Running   1 (103s ago)   106s
kubeflow                    metadata-grpc-deployment-5644fb9768-flzbz               2/2     Running   1 (103s ago)   110s
kube-system                 metrics-server-648b5df564-fjg7c                         1/1     Running   0              110s
kubeflow                    katib-mysql-77b9495867-jm48s                            1/1     Running   0              106s
kubeflow                    tensorboard-controller-deployment-658456ddf-bd6wp       3/3     Running   2 (102s ago)   106s
kubeflow                    training-operator-56c45d6567-q6c8b                      1/1     Running   0              109s
kubeflow                    katib-controller-75858c4ddf-2fbwj                       1/1     Running   3 (85s ago)    106s
kubeflow                    ml-pipeline-77b65d8669-nrz4v                            2/2     Running   3 (88s ago)    106s
kubeflow                    profiles-deployment-59cc895fbb-nhsbp                    3/3     Running   3 (88s ago)    110s
local-path-storage          local-path-provisioner-54fc6cbb55-nzzh6                 0/1     Error     4 (62s ago)    109s
```

일단 위의 수많은 명령을 통해 Kubeflow를 설치하긴 했으나,,,
설치 과정에서 Kuberenetes에 대한 실력이 너무나 부족한 것을 느껴 Kuberenetes부터 확실하게 공부해야겠다.

아래 명령어들을 통해 Ubuntu 서버에 K3s를 구축할 수 있다.

```bash
# Docker
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
$ sudo systemctl restart docker
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
$ newgrp docker

# Swap Memory
$ sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
$ sudo swapoff -a

# K3s Cluster
$ curl -sfL https://get.k3s.io | sh -s - --docker
$ sudo cat /etc/rancher/k3s/k3s.yaml
...
$ mkdir .kube
$ sudo cp /etc/rancher/k3s/k3s.yaml .kube/config
$ sudo chown $USER:$USER .kube/config

# Kubectl
$ curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
$ chmod +x kubectl
$ sudo mv kubectl /usr/local/bin/

# Helm
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh

# Kustomize
$ curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
$ sudo mv kustomize /usr/local/bin/kustomize
```

~~본격 삽질 시작~~

---

# References

1. [Kubeflow: Architecture](https://www.kubeflow.org/docs/started/architecture/)
2. [SAMSUNG SDS: Kubeflow](https://www.samsungsds.com/kr/insights/kubeflow.html)
3. [Kurly: Kurly만의 MLOps 구축하기 - 쿠브플로우 도입기](https://helloworld.kurly.com/blog/second-mlops/)
4. [BESPIN Global: Kubeflow 시작하기](https://www.bespinglobal.com/kubeflow-start/)