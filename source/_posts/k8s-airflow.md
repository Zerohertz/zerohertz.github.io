---
title: Apache Airflow Setup on Kubernetes
date: 2023-08-09 19:47:04
categories:
- 3. DevOps
tags:
- Airflow
- Kubernetes
- Traefik
- Home Server
---
# Introduction

열심히 구축한 [home server](https://zerohertz.github.io/home-server-init/)을 더 다양하게 활용하기 위해 (~~설치가 매우 간편한 K3s~~ $\rightarrow$ 많은 시행착오 끝에 K8s로,,,) K8s를 통해 Apache Airflow를 설치하고 배포해보겠다.
`https://${DDNS}/airflow`에 서비스가 구동될 수 있도록 ingress도 설정해보겠다.

![thumbnail](/images/k8s-airflow/260274063-0ce33604-b019-4600-aa35-b20c14284947.gif)

~~삽질 끝에 복이 온다!~~

<!-- More -->

---

# Apache Airflow

우선 local에서 확인을 하고 ingress로 경로 설정을 진행한다.
Local에 Apache Airflow를 설치하기 위해 [Helm](https://helm.sh/)을 사용할 것인데 이는 Kuberenetes의 package manager이다.

<details>
<summary>
Helm에 대해 좀 더 자세한 설명
</summary>

> Helm의 주요 구성요소

+ Charts: Helm의 패키지로, 어플리케이션을 실행하는 데 필요한 모든 리소스 정의를 포함합니다. 차트는 디렉토리 구조로 되어 있으며, 여러 개의 YAML 파일로 구성됩니다.
+ Releases: 클러스터에 배포된 차트의 인스턴스입니다. 하나의 차트를 여러 번 또는 여러 버전으로 클러스터에 배포할 수 있습니다. 각 배포는 새로운 릴리즈가 됩니다.
+ Repositories: 공유 차트의 위치입니다. 공개 또는 사설 리포지토리를 설정하여 차트를 저장하고 공유할 수 있습니다.

> Helm의 주요 기능

+ 쉬운 관리: Helm을 사용하면, 복잡한 어플리케이션도 명령 한 줄로 쉽게 설치, 업그레이드, 제거할 수 있습니다.
+ 버전 관리: 다양한 버전의 차트를 저장하고 관리할 수 있으므로, 어플리케이션의 이전 버전으로 롤백하기 쉽습니다.
+ 커스터마이제이션: `values.yaml` 파일을 통해 패키지의 기본 설정을 쉽게 변경할 수 있습니다. 이를 통해 동일한 차트로 여러 환경과 조건에 맞게 배포할 수 있습니다.
+ 커뮤니티 지원: Helm Hub나 Artifact Hub 같은 곳에서 커뮤니티가 관리하는 수백 개의 준비된 차트를 찾을 수 있어, 많은 공통 어플리케이션을 쉽게 배포할 수 있습니다.

</details>

<br />

```shell
$ helm repo add apache-airflow https://airflow.apache.org
"apache-airflow" has been added to your repositories
$ helm repo update
Hang tight while we grab the latest from your chart repositories...
...Successfully got an update from the "apache-airflow" chart repository
Update Complete. ⎈Happy Helming!⎈
$ helm install airflow apache-airflow/airflow -f values.yaml -n airflow --create-namespace
```

여기서 `values.yaml`은 대표적으로 아래와 같은 변수들이 존재한다.
더 자세한 사항은 `helm show values apache-airflow/airflow > values.yaml` 명령어를 실행하여 확인하면 된다.

|Name|Mean|Default|
|:-:|:-:|:-:|
|`executor`|사용할 Airflow executor (예: CeleryExecutor)|`CeleryExecutor`|
|`airflow.image.repository`|Airflow Docker 이미지 저장소|`apache/airflow`|
|`airflow.image.tag`|Airflow Docker 이미지 태그|차트 버전과 일치|
|`airflow.config`|Airflow 설정 (airflow.cfg 내용)|`{}`|
|`web.replicas`|Airflow 웹 서버의 레플리카 수|`1`|
|`scheduler.replicas`|Airflow 스케줄러의 레플리카 수|`1`|
|`dags.gitSync.enabled`|DAGs의 git-sync 활성화 여부|`false`|
|`dags.path`|DAGs 파일의 경로|`/opt/airflow/dags`|
|`postgresql.enabled`|내장 PostgreSQL 데이터베이스 사용 여부|`true`|
|`redis.enabled`|내장 Redis 사용 여부|`true`|
|`service.type`|Kubernetes 서비스 타입 (예: ClusterIP, NodePort)|`ClusterIP`|
|`ingress.enabled`|Ingress 리소스 활성화 여부|`false`|

## 삽질 그리고 삽질...

<details>
<summary>
삽질 !~
</summary>

```bash
Error: INSTALLATION FAILED: failed post-install: 1 error occurred:
        * timed out waiting for the condition
```

는 위와 같이 설치가 자꾸 안돼서 다른 버전을 선택해보려고 한다.

```shell
$ helm search repo apache-airflow/airflow --versions
NAME                    CHART VERSION   APP VERSION     DESCRIPTION                                       
apache-airflow/airflow  1.10.0          2.6.2           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.9.0           2.5.3           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.8.0           2.5.1           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.7.0           2.4.1           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.6.0           2.3.0           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.5.0           2.2.4           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.4.0           2.2.3           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.3.0           2.2.1           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.2.0           2.1.4           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.1.0           2.1.2           The official Helm chart to deploy Apache Airflo...
apache-airflow/airflow  1.0.0           2.0.2           Helm chart to deploy Apache Airflow, a platform...
$ helm install airflow apache-airflow/airflow --version ${Chart_Version} -n airflow --create-namespace
```

이마저도 안돼서 다른 방법을 찾아보던 중 [User-Community Airflow Helm Chart](https://artifacthub.io/packages/helm/airflow-helm/airflow)을 발견해서 아래와 같이 시도했다.

```shell
$ helm repo add airflow-stable https://airflow-helm.github.io/charts
"airflow-stable" has been added to your repositories
$ helm repo update
Hang tight while we grab the latest from your chart repositories...
...Successfully got an update from the "apache-airflow" chart repository
...Successfully got an update from the "airflow-stable" chart repository
Update Complete. ⎈Happy Helming!⎈
$ helm search repo airflow-stable/airflow --versions
NAME                    CHART VERSION   APP VERSION     DESCRIPTION                                       
airflow-stable/airflow  8.7.1           2.5.3           Airflow Helm Chart (User Community) - the stand...
airflow-stable/airflow  8.7.0           2.5.3           Airflow Helm Chart (User Community) - the stand...
airflow-stable/airflow  8.6.1           2.2.5           Airflow Helm Chart (User Community) - the stand...
airflow-stable/airflow  8.6.0           2.2.5           Airflow Helm Chart (User Community) - the stand...
airflow-stable/airflow  8.5.3           2.1.4           Airflow Helm Chart (User Community) - used to d...
airflow-stable/airflow  8.5.2           2.1.2           Airflow Helm Chart (User Community) - used to d...
airflow-stable/airflow  8.5.1           2.1.2           Airflow Helm Chart (User Community) - used to d...
airflow-stable/airflow  8.5.0           2.1.2           Airflow Helm Chart (User Community) - used to d...
airflow-stable/airflow  8.4.1           2.1.1           the community Apache Airflow Helm Chart - used ...
airflow-stable/airflow  8.4.0           2.1.1           the community Apache Airflow Helm Chart - used ...
airflow-stable/airflow  8.3.2           2.0.1           the community-maintained descendant of the stab...
airflow-stable/airflow  8.3.1           2.0.1           the community-maintained descendant of the stab...
airflow-stable/airflow  8.3.0           2.0.1           the community-maintained descendant of the stab...
airflow-stable/airflow  8.2.0           2.0.1           the community-maintained descendant of the stab...
airflow-stable/airflow  8.1.3           2.0.1           the community-maintained descendant of the stab...
airflow-stable/airflow  8.1.2           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.1.1           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.1.0           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.9           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.8           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.7           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.6           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.5           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.4           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.3           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.2           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.1           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  8.0.0           2.0.1           airflow is a platform to programmatically autho...
airflow-stable/airflow  7.16.0          1.10.12         airflow is a platform to programmatically autho...
airflow-stable/airflow  7.15.0          1.10.12         airflow is a platform to programmatically autho...
airflow-stable/airflow  7.14.3          1.10.12         airflow is a platform to programmatically autho...
airflow-stable/airflow  7.14.2          1.10.12         airflow is a platform to programmatically autho...
airflow-stable/airflow  7.14.1          1.10.12         airflow is a platform to programmatically autho...
airflow-stable/airflow  7.14.0          1.10.12         airflow is a platform to programmatically autho...
$ helm install airflow airflow-stable/airflow -n airflow --create-namespace
```

도저히 안된다...
아래 로그에서 발생하는 문제로 추정되는데 어디 부분인지 파악이 되지 않아 일단,,, 포기,,,

```bash
Readiness probe errored: rpc error: code = Unknown desc = http: invalid Host header
Liveness probe errored: rpc error: code = Unknown desc = http: invalid Host header
```

+ K3s $\rightarrow$ K8s
+ Ubuntu:22.04 $\rightarrow$ Ubuntu:20.04

위의 시도로 해결하려했으나,,, 모두 원점이다.

</details>

## Minikube? Kind?

[이 issue](https://github.com/apache/airflow/issues/29969#issuecomment-1540973564)를 보면 K8s 1.23+ 버전에서 위 오류가 지속적으로 발생한다는 의견이 있어서, K3s를 제외한 여러 환경에서 테스트해봤다.
[Minikube](https://kubernetes.io/ko/docs/tutorials/hello-minikube/)나 [Kind](https://kind.sigs.k8s.io/)로 설치하면 잘 되는 것을 확인할 수 있었다.

```shell
$ curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
$ chmod +x ./kind
$ sudo mv ./kind /usr/local/bin/kind
```

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
    kubeadmConfigPatches:
      -|
        kind: JoinConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "node=worker_0"
  - role: worker
    kubeadmConfigPatches:
      -|
        kind: JoinConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "node=worker_1"
    extraMounts:
      - hostPath: ./data
        containerPath: /tmp/data
  - role: worker
    kubeadmConfigPatches:
      -|
        kind: JoinConfiguration
        nodeRegistration:
          kubeletExtraArgs:
          node-labels: "node=worker_2"
    extraMounts:
      - hostPath: ./data
        containerPath: /tmp/data
```

```shell
$ kind create cluster --config kind.yaml --name airflow
$ helm install airflow apache-airflow/airflow -n airflow --create-namespace
```

```shell
$ kind delete cluster --name=airflow
$ sudo rm /usr/local/bin/kind
```

하지만 Minikube와 Kind는 내가 원하는 환경 구축에 적절하지 않아 Kubeadm을 이용해 구축하려 했다.
하지만... Ubuntu 22.04 LTS에 호완이 되지 않는 것으로 보였다. (~~아닐 수 있음...~~)
따라서 Ubuntu 20.04 LTS로 downgrade 후 1.22.8의 K8s를 설치했다.

## 결국 해낸 방법

Minikue, K3s, Kind와는 다르게 K8s를 사용하여 바로 `helm`으로 Airflow를 설치하려고 시도한다면 아래와 같은 이슈가 발생한다.

![Pending](/images/k8s-airflow/260209901-0ed8fa25-1a60-40d8-8e04-c1a208082dcc.png)

이 이유는 Pod가 bound할 `PersistentVolumeClaims`가 존재하지 않기 때문이다.
설명하기 앞서 `PVC`가 무엇인지 알아보자.

+ `PersistentVolume` (`PV`)
   + Cluster 내에 provisioning 된 저장 공간
   + 특정 저장 용량과 접근 모드 (`ReadWriteOnce`, `ReadOnlyMany`, `ReadWriteMany`) 존재
   + 수동으로 관리자에 의해 생성되거나, `StorageClass`를 사용하여 동적 provisioning 가능
+ `PersistentVolumeClaim` (`PVC`)
  + 사용자의 요청에 의해 생성, `PV`의 저장 용량과 같은 특정 양의 저장 공간을 요청하는 방법 제공
  + 사용자는 특정 저장 시스템이나 세부 구현에 대해 걱정할 필요 없이 저장 공간 요청
  + 일반적으로 특정 크기와 접근 모드를 요구하며, 이를 만족하는 PV가 연결
  + 적절한 `PV`가 없는 경우, `StorageClass`를 사용하여 동적으로 `PV`를 생성 가능
+ `StorageClass`
  + 동적 volume provisioning을 위한 정의
  + `PVC`가 생성되었을 때, 그에 맞는 `PV`가 없다면 `StorageClass`를 사용하여 자동으로 `PV` 생성
  + 이를 통해 특정 종류의 storage (AWS의 EBS, Google Cloud의 Persistent Disk, Azure의 Azure Disk Storage 등)를 provisioning 할 수 있는 방법과 파라미터 정의

쉽게 말해 위 오류는 `helm`으로 Airflow를 배포할 때 동적 provisioning이 되지 않기 때문에 발생한 현상이다.
따라서 적절한 `StorageClass`를 정의해야하며, 이를 위해 [local-path-provisioner](https://github.com/rancher/local-path-provisioner)를 사용했다.
설치와 setup은 아래와 같다.

```shell
$ kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
```

```yaml storage.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-path
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: rancher.io/local-path
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
```

![tada](/images/k8s-airflow/260212777-797b8896-f56e-4882-911e-bca3c48d5f71.gif)

이렇게 설치가 아주 잘 되는 것을 확인할 수 있다.
하지만 `dags` 폴더를 따로 연결하지 않아 DAG를 사용할 수 없는 상태이기 때문에 아래와 같은 `StorageClass`, `PV`, `PVC`를 생성한다.

```yaml storage.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: airflow-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-local-dags-pv
  labels:
    type: airflow-dags
spec:
  storageClassName: airflow-storage
  capacity:
    storage: 1Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: "${PATH}/dags"
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-local-dags-folder
  namespace: airflow
  annotations: {}
spec:
  storageClassName: airflow-storage
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 1Gi
  selector:
    matchLabels:
      type: airflow-dags
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: airflow-local-logs-pv
  labels:
    type: airflow-logs
spec:
  storageClassName: airflow-storage
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "${PATH}/logs"
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: airflow-local-logs-folder
  namespace: airflow
  annotations: {}
spec:
  storageClassName: airflow-storage
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  selector:
    matchLabels:
      type: airflow-logs
```

DAG가 저장될 `PV`와 `PVC`의 `accessModes`는 `ReadOnlyMany`로, log가 저장될 `PV`와 `PVC`의 `accessModes`는 `ReadWriteMany`로 정의해야한다.

<details>
<summary>
이유
</summary>

+ DAGs: `ReadOnlyMany`
  + 다중 접근: Airflow의 여러 컴포넌트 (스케줄러, 웹서버, 워커)가 동시에 DAGs를 읽을 필요가 있습니다. 따라서 여러 노드에서 동시에 해당 볼륨에 접근할 수 있어야 합니다.
  + 읽기 전용: 일반적으로 DAG 파일들은 실행 중에 변경되거나 쓰여지지 않습니다. 대신에 사용자나 CI/CD 파이프라인에 의해 업데이트 될 수 있습니다. 그러나 이러한 업데이트는 별도의 프로세스를 통해 이루어지며, 실행 중인 Airflow 서비스 자체에서는 DAG 파일을 수정할 필요가 없습니다. 따라서, 이 볼륨은 읽기 전용(`ReadOnlyMany`)으로 설정될 수 있습니다.
+ Logs: `ReadWriteMany`
  + 다중 접근 및 쓰기 가능: 워커 노드에서는 태스크의 실행 로그를 작성하고 저장해야 하며, 웹 서버는 사용자가 해당 로그를 볼 수 있도록 제공합니다. `KubernetesExecutor`를 사용하는 경우 각 태스크는 별도의 파드에서 실행됩니다. 따라서 여러 파드가 동시에 로그 볼륨에 쓰기 작업을 수행할 수 있어야 합니다.
  + 쓰기 및 읽기: 로그는 실행 중에 기록되므로 쓰기 작업이 필요합니다. 또한, 로그를 검토하거나 디버깅할 때 읽을 수 있어야 합니다. 따라서, 이 볼륨은 다중 접근 및 읽기/쓰기 모드(`ReadWriteMany`)로 설정되어야 합니다.
</details>
<br />

```shell
$ kubectl apply -f storageclass.yaml
$ kubectl get pv,pvc,storageclass -n airflow 
NAME                                                        CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                               STORAGECLASS         REASON   AGE
persistentvolume/airflow-local-dags-pv                      1Gi        ROX            Retain           Bound    airflow/airflow-local-dags-folder   airflow-storage               9m44s
persistentvolume/airflow-local-logs-pv                      10Gi       RWX            Retain           Bound    airflow/airflow-local-logs-folder   airflow-storage               9m44s
persistentvolume/pvc-40d5b2fc-4495-471b-a1f3-6ca1c4d3c478   8Gi        RWO            Delete           Bound    airflow/data-airflow-postgresql-0   local-path                    9m30s

NAME                                              STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS      AGE
persistentvolumeclaim/airflow-local-dags-folder   Bound    airflow-local-dags-pv                      1Gi        ROX            airflow-storage   9m44s
persistentvolumeclaim/airflow-local-logs-folder   Bound    airflow-local-logs-pv                      10Gi       RWX            airflow-storage   9m44s
persistentvolumeclaim/data-airflow-postgresql-0   Bound    pvc-40d5b2fc-4495-471b-a1f3-6ca1c4d3c478   8Gi        RWO            local-path        9m39s

NAME                                               PROVISIONER                    RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
storageclass.storage.k8s.io/airflow-storage        kubernetes.io/no-provisioner   Delete          WaitForFirstConsumer   false                  9h
```

![dags](/images/k8s-airflow/260214073-1652586e-9149-41d0-adea-e761d0f9f6a6.gif)

이렇게 `dags` 폴더도 잘 연결된다!

---

# Ingress

외부에 서비스를 배포하기 전 보안을 위해 아래와 같이 `values.yaml`을 수정한다.

```yaml values.yaml
# ----------------------- WEBSERVER ----------------------- #
webserver:
  defaultUser:
    enabled: true
    role: Admin
    username: ${USERNAME}
    email: ${EMAIL}
    firstName: ${FIRSTNAME}
    lastName: ${LASTNAME}
    password: ${PASSWORD}
# ----------------------- POSTGRESQL ----------------------- #
data:
  metadataConnection:
    user: ${USERNAME}
    pass: ${PASSWORD}
    protocol: postgresql
    host: ~
    port: 5432
    db: postgres
    sslmode: disable
postgresql:
  enabled: true
  image:
    tag: "11"
  auth:
    enablePostgresUser: true
    postgresPassword: ${POSTGRESPASSWORD}
    username: ${USERNAME}
    password: ${PASSWORD}
```

<img width="271" alt="404" src="/images/k8s-airflow/260240585-e995c1c1-037f-48a1-8e26-b2e8cb3ef66d.png">

`values.yaml`에 존재하는 `ingress.web.enabled`를 단순히 `true`로 설정하면 잘 실행되지만 목표인 `${DDNS}/airflow`를 달성하기 위해서 `false`로 설정하고 아래 `IngressRoute`를 추가한다! (~~위 사진은 무한한 삽질의 증거...~~)

```yaml traefik.yaml
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: airflow-webserver
  namespace: airflow
spec:
  stripPrefix:
    prefixes:
    - "/airflow"
---
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: airflow-webserver
  namespace: airflow
spec:
  entryPoints:
  - websecure
  routes:
  - match: Host(`${DDNS}`) && PathPrefix(`/airflow`)
    kind: Rule
    middlewares:
    - name: airflow-webserver
    services:
    - name: airflow-webserver
      port: 8080
  tls:
    certResolver: ${RESOLVER}
```

![ingress](/images/k8s-airflow/260254339-dcf92f50-7989-4b0e-b6b8-3a83cac62027.png)

짜자잔~ 성공... HTTPS가 빨간색인 이유는 일주일에 5번을 초과하게 인증서를 발급받아 생긴 현상이다... (~~무한 삽질~~)

---

# `KubernetesExecutor`

K8s로 설치를 완료했으니 여러 executor 또한 사용할 수 있다.

|Name|Definition|Pros|Cons|
|:-:|:-:|:-|:-|
|`SequentialExecutor`|로컬에서 순차적 실행|👍 설정 간단<br />👍 디버깅에 적합|👎 병렬 처리 미지원|
|`LocalExecutor`|로컬 별도 프로세스에서 병렬 실행|👍 로컬에서 병렬 처리 지원|👎 분산 환경 미지원|
|`CeleryExecutor`|Celery로 워커 노드에 분산 실행|👍 대규모 워크플로우와 병렬 처리 적합|👎 Celery 및 브로커 설정 필요|
|`DaskExecutor`|Dask로 작업 실행|👍 동적 워커 확장 가능|👎 Dask 설정 필요|
|`KubernetesExecutor`|작업을 Kubernetes 파드로 실행|👍 동적 리소스 할당<br />👍 독립적 환경 제공|👎 Kubernetes 설정 필요|
|`CeleryKubernetesExecutor`|Celery와 Kubernetes의 조합|👍 두 실행기의 장점 병합|👎 두 실행기 설정 및 유지 관리 필요|

`KubernetesExecutor`로 실행하면 아래와 같이 DAG 실행 시 pod가 생성되고 소멸하는 것을 확인할 수 있다.

![KubernetesExecutor](/images/k8s-airflow/260274063-0ce33604-b019-4600-aa35-b20c14284947.gif)

여기서 유의할 점은 `CeleryExecutor` 대신 `KubernetesExecutor`을 사용할 경우 아래와 같은 변경점이 있다.

+ `Redis`
  + `CeleryExecutor`에서는 메시지 브로커로 Redis나 RabbitMQ를 사용하여 여러 워커에 태스크를 분산합니다.
  + 그러나 `KubernetesExecutor`에서는 각 태스크를 별도의 Kubernetes Pod로 실행하기 때문에 `Redis`와 같은 메시지 브로커가 필요하지 않습니다.
+ `Worker`
  + `KubernetesExecutor`에서는 전통적인 Airflow Worker가 아닌 Kubernetes Pod를 사용하여 태스크를 실행합니다.
  + 그러므로 별도의 Airflow Worker 배포는 필요하지 않습니다.
+ `Triggerer`
  + Airflow 2.x의 일부로 도입된 `Triggerer`는 비동기 트리거 방식의 DAG를 지원하기 위해 사용됩니다.
  + `KubernetesExecutor`를 사용하면서 `Triggerer`를 필요로 하는 특정 기능을 사용하지 않는 한 `Triggerer`도 필요하지 않습니다.

만약 `CeleryKubernetesExecutor`를 사용하면 아래와 같이 지정하여 `CeleryExecutor`와 `KubernetesExecutor` 중 하나를 선택할 수 있다.

```python
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 8, 13),
}

with DAG('my_dag', default_args=default_args, schedule_interval=None) as dag:
    task_kubernetes = DummyOperator(
        task_id='task_kubernetes',
        executor_config={
            "KubernetesExecutor": {
                "image": "my-custom-image"
            }
        }
    )

    task_celery = DummyOperator(
        task_id='task_celery',
        executor_config={
            "CeleryExecutor": {
                "queue": "my_queue"
            }
        }
    )
```

---

# 읽으면 도움되는 것들

1. [쏘카: 쏘카 데이터 그룹 - Airflow와 함께한 데이터 환경 구축기(feat. Airflow on Kubernetes)](https://tech.socarcorp.kr/data/2021/06/01/data-engineering-with-airflow.html)
2. [오늘의집: 버킷플레이스 Airflow 도입기](https://www.bucketplace.com/post/2021-04-13-%EB%B2%84%ED%82%B7%ED%94%8C%EB%A0%88%EC%9D%B4%EC%8A%A4-airflow-%EB%8F%84%EC%9E%85%EA%B8%B0/)