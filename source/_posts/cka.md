---
title: CKA (Certified Kubernetes Administrator)
date: 2024-05-05 21:30:45
categories:
- 3. DevOps
tags:
- Docker
- Kubernetes
---
# Introduction

> CKA: Kubernetes 관리자의 책임을 수행할 수 있는 기술, 지식 및 역량을 갖추고 있음을 보증하는 자격증

<img src="/images/cka/certificate.png" alt="certificate" width=700 />

[CKA curriculum](https://github.com/cncf/curriculum/blob/master/CKA_Curriculum_v1.29.pdf)에서 CKA가 포함하는 내용들을 아래와 같이 확인할 수 있다.

<div align="right"><code>v1.29</code> 기준</div>

|                         Domain                          | Weight | Key Points                                                                                                                                                                                                                                                                                                                                                                |
| :-----------------------------------------------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Cluster Architecture,<br />Installation & Configuration |  25%   | ✅ Manage role based access control (RBAC)<br />✅ Use Kubeadm to install a basic cluster<br />✅ Manage a highly-available Kubernetes cluster<br />✅ Provision underlying infrascruture to deploy a Kubernetes cluster<br />✅ Implement etcd backup and restore                                                                                                             |
|                 Workloads & Scheduling                  |  15%   | ✅ Understand deployments and how to perform rolling update and rollbacks<br />✅ Use ConfigMaps and Secrets to configure applications<br />✅ Know how to scale applications<br />✅ Understand how resource limits can effect Pod scheduling<br />✅ Awareness of manifest management and common templating tools                                                            |
|                  Services & Networking                  |  20%   | ✅ Understand host networking configuration on the cluster nodes<br />✅ Understand connectivity between Pods<br />✅ Understand ClusterIP, NodePort, LoadBalancer service types and endpoints<br />✅ Know how to use Ingress controllers and Ingress resources<br />✅ Know how to configure and use CoreDNS<br />✅ Choose an appropriate container network interface plugin |
|                         Storage                         |  10%   | ✅ Understand storage classes, persistent volumes<br />✅ Understand volume mode, access modes and reclaim polices for volumes<br />✅ Understand persistent volume claims primitive<br />✅ Know how to configure applications with persistent storage                                                                                                                       |
|                     Troubleshooting                     |  30%   | ✅ Evaluate cluster and node logging<br />✅ Understand how to monitor applications<br />✅ Manage container stdout & stderr logs<br />✅ Troubleshoot application failure<br />✅ Troubleshoot cluster component failure<br />✅ Troubleshoot networking                                                                                                                       |

<!-- More -->

+ 가격: \\$375
+ 시간: 2시간
+ 문제: 17문제
+ 장소: 사방이 막힌 조용한 장소
+ 준비물: 신분증 (영문 이름 필수)

[공식 사이트](https://training.linuxfoundation.org/certification/certified-kubernetes-administrator-cka/)에서 결제하여 CKA 응시를 신청할 수 있다.
결제 전에 [Coupert](https://chromewebstore.google.com/detail/coupert-automatic-coupon/mfidniedemcgceagapgdekdbmanojomk?pli=1)를 설치하면 기존 \\$395의 가격을 할인 받을 수 있다. (필자는 40%의 할인을 받아 \\$237에 결제했다.)
결제를 마쳤다면 1년 내로 아래와 같이 시험을 예약해야 한다.

![cka-exam-date](/images/cka/cka-exam-date.png)

[시험 응시 시 환경](https://docs.linuxfoundation.org/tc-docs/certification/tips-cka-and-ckad#exam-technical-instructions)에서는 현재 존재하지 않지만 multi-cluster 환경에서 시험을 응시하게 된다.
[여기](https://syscheck.bridge.psiexams.com/)에서 시험 응시 시 사용할 기기의 검증을 수행할 수 있다.
[Udemy에서 Mumshad님이 진행하신 강의](https://www.udemy.com/course/certified-kubernetes-administrator-with-practice-tests/)가 매우 유명하기 때문에 해당 강의를 수강했다.
해당 강의를 수강하면 [KodeKloud](https://kodekloud.com/courses/certified-kubernetes-administrator-cka/)를 통해 실제 시험과 유사한 조건 속에서 연습할 수 있다.
마지막으로 CKA 시험의 결제를 마치면 아래와 같이 [killer.sh](https://killer.sh/)의 문제를 2회 풀 수 있는 권한을 주기 때문에 복기를 위해 이를 풀었다.

![killer.sh](/images/cka/killer.sh.png)

---

# Theoretical Backgrounds

## Kubernetes Components

[Kubernetes의 요소](https://kubernetes.io/docs/concepts/architecture/)들은 아래와 같이 구성된다.

![schematic](/images/cka/schematic.png)

## Core Concepts

<details>
<summary>
Pods
</summary>

```shell
$ kubectl get po | wc -l
No resources found in default namespace.
0
$ kubectl run nginx --image nginx
pod/nginx created
$ kubectl get po | wc -l
5
$ kubectl describe po | grep Image:
    Image:          nginx
    Image:         busybox
    Image:         busybox
    Image:         busybox
$ kubectl get po -owide
NAME            READY   STATUS    RESTARTS   AGE    IP           NODE           NOMINATED NODE   READINESS GATES
nginx           1/1     Running   0          2m     10.42.0.9    controlplane   <none>           <none>
newpods-dk2w8   1/1     Running   0          111s   10.42.0.11   controlplane   <none>           <none>
newpods-gr9fm   1/1     Running   0          111s   10.42.0.12   controlplane   <none>           <none>
newpods-p4r2q   1/1     Running   0          111s   10.42.0.10   controlplane   <none>           <none>
$ kubectl get po webapp
NAME     READY   STATUS             RESTARTS   AGE
webapp   1/2     ImagePullBackOff   0          24s
$ kubectl describe po webapp | grep Image:
    Image:          nginx
    Image:          agentx
$ kubectl describe po webapp | grep agentx -A 10 | grep -i state
    State:          Waiting
$ kubectl delete po webapp
pod "webapp" deleted
$ kubectl run redis --image redis123
pod/redis created
$ kubectl edit po redisf
...
spec:
  containers:
    - image: redis
      imagePullPolicy: Always
      name: redis
...
pod/redis edited
```

</details>

<details>
<summary>
ReplicaSets
</summary>

```shell
$ kubectl get po | wc -l
No resources found in default namespace.
0
$ kubectl get rs | wc -l
No resources found in default namespace.
0
$ kubectl get rs | wc -l
2
$ kubectl get rs new-replica-set
NAME              DESIRED   CURRENT   READY   AGE
new-replica-set   4         4         0       25s
$ kubectl describe rs new-replica-set | grep Image:
    Image:      busybox777
$ kubectl get po
NAME                    READY   STATUS             RESTARTS   AGE
new-replica-set-v85ct   0/1     ImagePullBackOff   0          90s
new-replica-set-985cd   0/1     ImagePullBackOff   0          90s
new-replica-set-hngqj   0/1     ImagePullBackOff   0          90s
new-replica-set-7gxqp   0/1     ImagePullBackOff   0          90s
$ vi replicaset-definition-1.yaml
apiVersion: apps/v1
...
$ kubectl apply -f replicaset-definition-1.yaml
replicaset.apps/replicaset-1 created
$ vi replicaset-definition-2.yaml
...
spec:
  replicas: 2
  selector:
    matchLabels:
      tier: nginx
...
$ kubectl apply -f replicaset-definition-2.yaml
replicaset.apps/replicaset-2 created
$ kubectl delete rs replicaset-1
replicaset.apps "replicaset-1" deleted
$ kubectl delete rs replicaset-2
replicaset.apps "replicaset-2" deleted
$ vi tmp.yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: new-replica-set
spec:
  replicas: 4
  selector:
    matchLabels:
      type: busybox
  template:
    metadata:
      labels:
        type: busybox
    spec:
      containers:
        - name: busybox
          image: busybox
          command:
            - sh
            - -c
            - echo Hello Kubernetes && sleep 3600
$ kubectl apply -f tmp.yaml
replicaset.apps/new-replica-set created
$ kubectl scale rs new-replica-set --replicas 5
replicaset.apps/new-replica-set scaled
$ kubectl scale rs new-replica-set --replicas 2
replicaset.apps/new-replica-set scaled
```

</details>

<details>
<summary>
Deployments
</summary>

```shell
$ kubectl get po
No resources found in default namespace.
$ kubectl get rs
No resources found in default namespace.
$ kubectl get deploy
No resources found in default namespace.
$ kubectl get deploy
NAME                  READY   UP-TO-DATE   AVAILABLE   AGE
frontend-deployment   0/4     4            0           7s
$ kubectl get rs
NAME                             DESIRED   CURRENT   READY   AGE
frontend-deployment-7b9984b987   4         4         0       21s
$ kubectl get po
NAME                                   READY   STATUS             RESTARTS   AGE
frontend-deployment-7b9984b987-84kqx   0/1     ImagePullBackOff   0          30s
frontend-deployment-7b9984b987-z9qhm   0/1     ImagePullBackOff   0          30s
frontend-deployment-7b9984b987-zsg7l   0/1     ErrImagePull       0          30s
frontend-deployment-7b9984b987-6zstm   0/1     ErrImagePull       0          30s
$ kubectl describe deploy | grep Image
    Image:      busybox888
$ vi deployment-definition-1.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-1
spec:
  replicas: 2
  selector:
    matchLabels:
      name: busybox-pod
  template:
    metadata:
      labels:
        name: busybox-pod
    spec:
      containers:
        - name: busybox-container
          image: busybox888
          command:
            - sh
            - "-c"
            - echo Hello Kubernetes! && sleep 3600
$ kubectl apply -f deployment-definition-1.yaml
deployment.apps/deployment-1 created
$ kubectl create deploy httpd-frontend --image httpd:2.4-alpine
deployment.apps/httpd-frontend created
$ kubectl scale deploy httpd-frontend --replicas 3
deployment.apps/httpd-frontend scaled
```

</details>

<details>
<summary>
Namespaces
</summary>

```shell
$ kubectl get ns | wc -l
11
$ kubectl get -n research po
NAME    READY   STATUS      RESTARTS      AGE
dna-1   0/1     Completed   2 (24s ago)   35s
dna-2   0/1     Completed   2 (23s ago)   35s
$ kubectl run -n finance redis --image redis
pod/redis created
$ kubectl get -A po | grep blue
marketing       blue                                      1/1     Running            0             109s
$ kubectl get -n marketing svc
NAME           TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
blue-service   NodePort   10.43.254.241   <none>        8080:30082/TCP   3m41s
db-service     NodePort   10.43.229.251   <none>        6379:32247/TCP   3m41s
```

</details>

<details>
<summary>
Services
</summary>

```shell
$ kubectl get svc
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.43.0.1    <none>        443/TCP   4m25s
$ kubectl describe svc | grep Target
TargetPort:        6443/TCP
$ kubectl describe svc | grep -i labels -A1
Labels:            component=apiserver
                   provider=kubernetes
$ kubectl describe svc | grep -i end
Endpoints:         192.23.255.9:6443
$ kubectl get deploy
NAME                       READY   UP-TO-DATE   AVAILABLE   AGE
simple-webapp-deployment   4/4     4            4           11s
$ kubectl describe deploy | grep Image
    Image:        kodekloud/simple-webapp:red
$ vi service-definition-1.yaml
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
  namespace: default
spec:
  ports:
    - nodePort: 30080
      port: 8080
      targetPort: 8080
  selector:
    name: simple-webapp
  type: NodePort
$ kubectl apply -f service-definition-1.yaml
service/webapp-service created
```

</details>

<details>
<summary>
Imperative Commands
</summary>

```shell
$ kubectl run nginx-pod --image nginx:alpine
pod/nginx-pod created
$ kubectl run redis -l tier=db --image redis:alpine
pod/redis created
$ kubectl expose po redis --name redis-service --port 6379
service/redis-service exposed
$ kubectl create deploy webapp --image kodekloud/webapp-color --replicas 3
deployment.apps/webapp created
$ kubectl run custom-nginx --image nginx --port 8080
pod/custom-nginx created
$ kubectl create ns dev-ns
namespace/dev-ns created
$ kubectl create -n dev-ns deploy redis-deploy --image redis --replicas 2
deployment.apps/redis-deploy created
$ kubectl run httpd --image httpd:alpine
pod/httpd created
$ kubectl expose po httpd --name httpd --port 80
service/httpd exposed
```

</details>

## Scheduling

<details>
<summary>
Manual Scheduling
</summary>

```shell
$ kubectl apply -f nginx.yaml
pod/nginx created
$ kubectl get po
NAME    READY   STATUS    RESTARTS   AGE
nginx   0/1     Pending   0          13s
$ kubectl get po nginx -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  nodeName: node01
  containers:
    - image: nginx
...
$ kubectl replace -f tmp.yaml --force
pod "nginx" deleted
pod/nginx replaced
$ vi tmp.yaml
...
spec:
  nodeName: controlplane
  containers:
    - image: nginx
...
$ kubectl replace -f tmp.yaml --force
pod "nginx" deleted
pod/nginx replaced
```

</details>

<details>
<summary>
Labels and Selectors
</summary>

```shell
$ kubectl describe po | grep dev | wc -l
7
$ kubectl get po -l bu=finance | wc -l
7
$ kubectl get all -l env=prod --no-headers | wc -l
7
$ kubectl get po -l env=prod,bu=finance,tier=frontend
NAME          READY   STATUS    RESTARTS   AGE
app-1-zzxdf   1/1     Running   0          4m51s
$ vi replicaset-definition-1.yaml
...
spec:
  replicas: 2
  selector:
    matchLabels:
      tier: nginx
  template:
    metadata:
      labels:
        tier: nginx
...
$ kubectl apply -f replicaset-definition-1.yaml
replicaset.apps/replicaset-1 created
```

</details>

<details>
<summary>
Taints and Tolerations
</summary>

```shell
$ kubectl get node
NAME           STATUS   ROLES           AGE     VERSION
controlplane   Ready    control-plane   2m58s   v1.29.0
node01         Ready    <none>          2m20s   v1.29.0
$ kubectl taint node node01 spray=mortein:NoSchedule
node/node01 tainted
$ kubectl run mosquito --image nginx
pod/mosquito created
$ kubectl get po
NAME       READY   STATUS    RESTARTS   AGE
mosquito   0/1     Pending   0          100s
$ kubectl run bee --image nginx --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    run: bee
  name: bee
spec:
  containers:
    - image: nginx
      name: bee
      resources: {}
  tolerations:
    - key: spray
      value: mortein
      effect: NoSchedule
      operator: Equal
$ kubectl apply -f tmp.yaml
pod/bee created
$ kubectl describe node controlplane | grep -i taints
Taints:             node-role.kubernetes.io/control-plane:NoSchedule
$ kubectl taint node controlplane node-role.kubernetes.io/control-plane:NoSchedule-
node/controlplane untainted
$ kubectl get po -owide
NAME       READY   STATUS    RESTARTS   AGE     IP           NODE           NOMINATED NODE   READINESS GATES
bee        1/1     Running   0          4m42s   10.244.1.2   node01         <none>           <none>
mosquito   1/1     Running   0          12m     10.244.0.4   controlplane   <none>           <none>
```

</details>

<details>
<summary>
Node Affinity
</summary>

```shell
$ kubectl describe node node01 | grep -i labels -A4
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=node01
                    kubernetes.io/os=linux
$ kubectl describe node node01 | grep arch
Labels:             beta.kubernetes.io/arch=amd64
                    kubernetes.io/arch=amd64
$ kubectl label node node01 color=blue
node/node01 labeled
$ kubectl create deploy blue --image nginx
deployment.apps/blue created
$ kubectl scale deploy blue --replicas 3
deployment.apps/blue scaled
$ kubectl get po -owide
NAME                    READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
blue-747bd9c977-gsslx   1/1     Running   0          26s   10.244.1.3   node01         <none>           <none>
blue-747bd9c977-l426t   1/1     Running   0          26s   10.244.0.4   controlplane   <none>           <none>
blue-747bd9c977-ztsc2   1/1     Running   0          34s   10.244.1.2   node01         <none>           <none>
$ kubectl edit deploy blue
...
spec:
...
  template:
...
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: color
                    operator: In
                    values:
                      - blue
...
deployment.apps/blue edited
$ kubectl get po -owide
NAME                   READY   STATUS    RESTARTS   AGE   IP           NODE     NOMINATED NODE   READINESS GATES
blue-8b4fdbcb5-58nt7   1/1     Running   0          39s   10.244.1.4   node01   <none>           <none>
blue-8b4fdbcb5-5fcjk   1/1     Running   0          15s   10.244.1.5   node01   <none>           <none>
blue-8b4fdbcb5-5wwbj   1/1     Running   0          8s    10.244.1.6   node01   <none>           <none>
$ kubectl create deploy red --image nginx --replicas 2 --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
...
  template:
...
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: node-role.kubernetes.io/control-plane
                    operator: Exists
...
$ kubectl apply -f tmp.yaml
deployment.apps/red created
```

</details>

<details>
<summary>
Resource Limits
</summary>

```shell
$ kubectl describe po | grep cpu: -B1
    Limits:
      cpu:  2
    Requests:
      cpu:        1
$ kubectl delete po rabbit
pod "rabbit" deleted
$ kubectl describe po elephant | grep State -A2
    State:          Waiting
      Reason:       CrashLoopBackOff
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    1
$ kubectl get po elephant -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  containers:
    - args:
...
      resources:
        limits:
          memory: 20Mi
...
$ kubectl replace -f tmp.yaml --force
pod "elephant" deleted
pod/elephant replaced
$ kubectl delete po elephant
pod "elephant" deleted
```

</details>

<details>
<summary>
DaemonSets
</summary>

```shell
$ kubectl get -A ds
NAMESPACE      NAME              DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR            AGE
kube-flannel   kube-flannel-ds   1         1         1       1            1           <none>                   39s
kube-system    kube-proxy        1         1         1       1            1           kubernetes.io/os=linux   40s
$ kubectl describe -n kube-system ds | grep Pods
Number of Nodes Scheduled with Up-to-date Pods: 1
Number of Nodes Scheduled with Available Pods: 1
Pods Status:  1 Running / 0 Waiting / 0 Succeeded / 0 Failed
$ kubectl describe -n kube-flannel ds kube-flannel-ds | grep Image
    Image:      docker.io/flannel/flannel-cni-plugin:v1.2.0
    Image:      docker.io/flannel/flannel:v0.23.0
    Image:      docker.io/flannel/flannel:v0.23.0
$ kubectl create -n kube-system deploy elasticsearch --image registry.k8s.io/fluentd-elasticsearch:1.20 --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    app: elasticsearch
  name: elasticsearch
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
        - image: registry.k8s.io/fluentd-elasticsearch:1.20
          name: fluentd-elasticsearch
$ kubectl apply -f tmp.yaml
daemonset.apps/elasticsearch created
```

</details>

<details>
<summary>
Static Pods
</summary>

```shell
$ kubectl get -A po | grep controlplane
kube-system    etcd-controlplane                      1/1     Running   0          2m57s
kube-system    kube-apiserver-controlplane            1/1     Running   0          2m57s
kube-system    kube-controller-manager-controlplane   1/1     Running   0          2m57s
kube-system    kube-scheduler-controlplane            1/1     Running   0          3m
$ kubectl get -A po -owide | grep controlplane
kube-flannel   kube-flannel-ds-26lxf                  1/1     Running   0          3m31s   192.12.15.6   controlplane   <none>           <none>
kube-system    coredns-69f9c977-r7jbc                 1/1     Running   0          3m31s   10.244.0.3    controlplane   <none>           <none>
kube-system    coredns-69f9c977-r8jh9                 1/1     Running   0          3m31s   10.244.0.2    controlplane   <none>           <none>
kube-system    etcd-controlplane                      1/1     Running   0          3m42s   192.12.15.6   controlplane   <none>           <none>
kube-system    kube-apiserver-controlplane            1/1     Running   0          3m42s   192.12.15.6   controlplane   <none>           <none>
kube-system    kube-controller-manager-controlplane   1/1     Running   0          3m42s   192.12.15.6   controlplane   <none>           <none>
kube-system    kube-proxy-tk224                       1/1     Running   0          3m31s   192.12.15.6   controlplane   <none>           <none>
kube-system    kube-scheduler-controlplane            1/1     Running   0          3m45s   192.12.15.6   controlplane   <none>           <none>
$ ls /etc/kubernetes/manifests
etcd.yaml  kube-apiserver.yaml  kube-controller-manager.yaml  kube-scheduler.yaml
$ cat /etc/kubernetes/manifests/kube-apiserver.yaml | grep image
    image: registry.k8s.io/kube-apiserver:v1.29.0
    imagePullPolicy: IfNotPresent
$ kubectl run static-busybox --image busybox --dry-run=client -oyaml --command -- sleep 1000 > /etc/kubernetes/manifests/static-busybox.yaml
$ vi /etc/kubernetes/manifests/static-busybox.yaml
...
spec:
  containers:
    - command:
        - sleep
        - "1000"
      image: busybox:1.28.4
...
$ kubectl delete po static-busybox-controlplane
pod "static-busybox-controlplane" deleted
$ kubectl describe po static-busybox-controlplane | grep Image:
    Image:         busybox:1.28.4
$ kubectl get -A po | grep static-greenbox
default        static-greenbox-node01                 1/1     Running   0          58s
$ ssh node01
$ ps -ef | grep kubelet
root        9353       1  0 14:05 ?        00:00:01 /usr/bin/kubelet --bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf --kubeconfig=/etc/kubernetes/kubelet.conf --config=/var/lib/kubelet/config.yaml --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock --pod-infra-container-image=registry.k8s.io/pause:3.9
root       10444   10288  0 14:07 pts/0    00:00:00 grep kubelet
$ grep -i staticpod /var/lib/kubelet/config.yaml
staticPodPath: /etc/just-to-mess-with-you
$ rm /etc/just-to-mess-with-you/greenbox.yaml
$ kubectl get po
No resources found in default namespace.
```

</details>

<details>
<summary>
Multiple Schedulers
</summary>

```shell
$ kubectl get -n kube-system po | grep scheduler
kube-scheduler-controlplane            1/1     Running   0          5m9s
$ kubectl describe -n kube-system po kube-scheduler-controlplane | grep Image:
    Image:         registry.k8s.io/kube-scheduler:v1.29.0
$ kubectl apply -f my-scheduler-configmap.yaml
configmap/my-scheduler-config created
$ vi my-scheduler.yaml
...
spec:
  serviceAccountName: my-scheduler
  containers:
    - command:
        - /usr/local/bin/kube-scheduler
        - --config=/etc/kubernetes/my-scheduler/my-scheduler-config.yaml
      image: registry.k8s.io/kube-scheduler:v1.29.0
...
$ kubectl apply -f my-scheduler.yaml
pod/my-scheduler created
$ vi nginx-pod.yaml
...
spec:
  schedulerName: my-scheduler
...
$ kubectl apply -f nginx-pod.yaml
pod/nginx created
```

</details>

## Logging & Monitoring

<details>
<summary>
Monitor Cluster Components
</summary>

```shell
$ git clone https://github.com/kodekloudhub/kubernetes-metrics-server.git
$ cd kubernetes-metrics-server
$ kubectl apply -f .
clusterrole.rbac.authorization.k8s.io/system:aggregated-metrics-reader created
clusterrolebinding.rbac.authorization.k8s.io/metrics-server:system:auth-delegator created
rolebinding.rbac.authorization.k8s.io/metrics-server-auth-reader created
apiservice.apiregistration.k8s.io/v1beta1.metrics.k8s.io created
serviceaccount/metrics-server created
deployment.apps/metrics-server created
service/metrics-server created
clusterrole.rbac.authorization.k8s.io/system:metrics-server created
clusterrolebinding.rbac.authorization.k8s.io/system:metrics-server created
$ kubectl top node
NAME           CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
controlplane   298m         0%     1067Mi          0%
node01         167m         0%     275Mi           0%
$ kubectl top po
NAME       CPU(cores)   MEMORY(bytes)
elephant   16m          32Mi
lion       1m           18Mi
rabbit     98m          252Mi
```

</details>

<details>
<summary>
Managing Application Logs
</summary>

```shell
$ kubectl logs webapp-1 | grep USER5 | tail -n1
[2024-04-22 14:21:22,127] WARNING in event-simulator: USER5 Failed to Login as the account is locked due to MANY FAILED ATTEMPTS.
$ kubectl logs webapp-2
...
[2024-04-22 14:22:02,265] WARNING in event-simulator: USER30 Order failed as the item is OUT OF STOCK.
...
```

</details>

## Application Lifecycle Management

<details>
<summary>
Rolling Updates and Rollbacks
</summary>

```shell
$ kubectl get po
NAME                       READY   STATUS    RESTARTS   AGE
frontend-685dfcc44-ql4jt   1/1     Running   0          50s
frontend-685dfcc44-xzqm9   1/1     Running   0          50s
frontend-685dfcc44-v858j   1/1     Running   0          50s
frontend-685dfcc44-rtr5k   1/1     Running   0          50s
$ kubectl describe po | grep Image:
    Image:          kodekloud/webapp-color:v1
    Image:          kodekloud/webapp-color:v1
    Image:          kodekloud/webapp-color:v1
    Image:          kodekloud/webapp-color:v1
$ kubectl describe deploy | grep -i strategy
StrategyType:           RollingUpdate
RollingUpdateStrategy:  25% max unavailable, 25% max surge
$ kubectl edit deploy frontend
...
spec:
...
  template:
...
    spec:
      containers:
        - image: kodekloud/webapp-color:v2
...
deployment.apps/frontend edited
$ kubectl get deploy frontend -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
...
  strategy:
    type: Recreate
...
$ kubectl replace -f tmp.yaml --force
deployment.apps "frontend" deleted
deployment.apps/frontend replaced
$ kubectl edit deploy frontend
...
spec:
...
  template:
...
    spec:
      containers:
        - image: kodekloud/webapp-color:v3
...
deployment.apps/frontend edited
```

</details>

<details>
<summary>
Commands and Argumnets
</summary>

```shell
$ kubectl get po
NAME             READY   STATUS    RESTARTS   AGE
ubuntu-sleeper   1/1     Running   0          5s
$ kubectl describe po | grep -i comm -A2
    Command:
      sleep
      4800

$ vi ubuntu-sleeper-2.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ubuntu-sleeper-2
spec:
  containers:
    - name: ubuntu
      image: ubuntu
      command:
        - sleep
        - "5000"
$ kubectl apply -f ubuntu-sleeper-2.yaml
pod/ubuntu-sleeper-2 created
$ kubectl apply -f ubuntu-sleeper-3.yaml
pod/ubuntu-sleeper-3 created
$ kubectl replace -f ubuntu-sleeper-3.yaml --force
pod "ubuntu-sleeper-3" deleted
pod/ubuntu-sleeper-3 replaced
$ cat webapp-color/Dockerfile | tail -n1
ENTRYPOINT ["python", "app.py"]
$ cat webapp-color/Dockerfile2 | tail -n3
ENTRYPOINT ["python", "app.py"]

CMD ["--color", "red"]
$ cat webapp-color-2/webapp-color-pod.yaml | grep comm
    command: ["--color","green"]
$ cat webapp-color-3/webapp-color-pod-2.yaml | grep comm -A1
    command: ["python", "app.py"]
    args: ["--color", "pink"]
$ kubectl run webapp-green --image kodekloud/webapp-color --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    run: webapp-green
  name: webapp-green
spec:
  containers:
    - image: kodekloud/webapp-color
      name: webapp-green
      args: ["--color", "green"]
$ kubectl apply -f tmp.yaml
pod/webapp-green created
```

</details>

<details>
<summary>
Env Variables
</summary>

```shell
$ kubectl get po | wc -l
2
$ kubectl describe po | grep -i env -A1
    Environment:
      APP_COLOR:  pink
$ kubectl get po webapp-color -oyaml > tmp.yaml
$ cat tmp.yaml | grep -i env -A2
  - env:
    - name: APP_COLOR
      value: pink
$ sed -i 's/pink/green/g' tmp.yaml
$ cat tmp.yaml | grep -i env -A2
  - env:
    - name: APP_COLOR
      value: green
$ kubectl replace -f tmp.yaml --force
pod "webapp-color" deleted
pod/webapp-color replaced
$ kubectl get cm | wc -l
3
$ kubectl describe cm db-config | grep -i host -A2
DB_HOST:
----
SQL01.example.com
$ kubectl create cm webapp-config-map --from-literal APP_COLOR=darkblue --from-literal APP_OTHER=disregard
configmap/webapp-config-map created
$ vi tmp.yaml
...
spec:
  containers:
    - env:
        - name: APP_COLOR
          valueFrom:
            configMapKeyRef:
              name: webapp-config-map
              key: APP_COLOR
...
$ kubectl replace -f tmp.yaml --force
pod "webapp-color" deleted
pod/webapp-color replaced
```

</details>

<details>
<summary>
Secrets
</summary>

```shell
$ kubectl get secret | wc -l
2
$ kubectl describe secret dashboard-token
Name:         dashboard-token
Namespace:    default
Labels:       <none>
Annotations:  kubernetes.io/service-account.name: dashboard-sa
              kubernetes.io/service-account.uid: f4581f21-a83b-46b9-b5a2-35b09c74661b

Type:  kubernetes.io/service-account-token

Data
====
namespace:  7 bytes
token:      eyJhbGciOiJSUzI1NiIsImtpZCI6ImJtRUh3ZmZEMG15VTlKeUF1aWswX1NON2FmdkNwMXFhZ0w5WmZZWEhUcVkifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRhc2hib2FyZC10b2tlbiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJkYXNoYm9hcmQtc2EiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiJmNDU4MWYyMS1hODNiLTQ2YjktYjVhMi0zNWIwOWM3NDY2MWIiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpkYXNoYm9hcmQtc2EifQ.stnWhgZ4_CriV8HUE2lbKFGvUNrGBYNmX0GFHEBct3f5UXijVH_uL7zCJfhKsZCF2DOeOIn48AxQPyqszm9ZSzZeUtAp5pP1Oo1gfHnnmUyTHhTJlkjpnHHkugQyOBgfDtXn1V4fqVmLGFqiXrUx2-v_1xQZaVDtEBEe5r3dEDWYAbfZOoGtisso69iTWEh8_58JUqStoY43NIShlW4r6mgNyM8wA9xZxUSeGRsmtdQmrJLAJQQIHk_6KQ04mg11ijbUtGmX4r29b-aT68jpYx4K-b7K7SMEvYhjZV8rmm8BkmJdrGwivAl1r1F_w_Sg4vIK1qqbh1mwx84t1pysnA
ca.crt:     570 bytes
$ kubectl create secret generic db-secret --from-literal DB_Host=sql01 --from-literal DB_User=root --from-literal DB_Password=password123
secret/db-secret created
$ kubectl get po webapp-pod -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  containers:
    - envFrom:
        - secretRef:
            name: db-secret
...
$ kubectl replace -f tmp.yaml --force
pod "webapp-pod" deleted
pod/webapp-pod replaced
```

</details>

<details>
<summary>
Multi Container Pods
</summary>

```shell
$ kubectl get po red
NAME   READY   STATUS              RESTARTS   AGE
red    0/3     ContainerCreating   0          12s
$ kubectl describe po blue
...
Containers:
  teal:
...
  navy:
...
$ kubectl run yellow --image busybox --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: yellow
  name: yellow
spec:
  containers:
    - image: busybox
      name: lemon
      command:
        - sleep
        - "1000"
    - image: redis
      name: gold
$ kubectl apply -f tmp.yaml
pod/yellow created
$ kubectl get -n elastic-stack po app
NAME   READY   STATUS    RESTARTS   AGE
app    1/1     Running   0          7m15s
$ kubectl exec -n elastic-stack app -- cat /log/app.log | grep -i login | tail -n2
[2024-04-23 09:59:54,322] WARNING in event-simulator: USER5 Failed to Login as the account is locked due to MANY FAILED ATTEMPTS.
[2024-04-23 09:59:54,822] WARNING in event-simulator: USER5 Failed to Login as the account is locked due to MANY FAILED ATTEMPTS.
$ kubectl run -n elastic-stack app --image kodekloud/filebeat-configured --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: app
  name: app
  namespace: elastic-stack
spec:
  containers:
    - image: kodekloud/filebeat-configured
      name: sidecar
      volumeMounts:
        - mountPath: /var/log/event-simulator/
          name: log-volume
    - image: kodekloud/event-simulator
      name: app
      volumeMounts:
        - mountPath: /log
          name: log-volume
  volumes:
    - name: log-volume
      hostPath:
        path: /var/log/webapp
        type: DirectoryOrCreate
$ kubectl replace -f tmp.yaml --force
pod "app" deleted
pod/app replaced
```

</details>

<details>
<summary>
Init Containers
</summary>

```shell
$ kubectl describe po blue | grep -i init
Init Containers:
  init-myservice:
  Initialized                 True
  Normal  Created    51s   kubelet            Created container init-myservice
  Normal  Started    51s   kubelet            Started container init-myservice
$ kubectl describe po blue | grep -i init -A10 | grep Image:
    Image:         busybox
$ kubectl describe po blue | grep -i init -A10 | grep -i state
    State:          Terminated
$ kubectl describe po purple
...
Init Containers:
  warm-up-1:
...
  warm-up-2:
...
$ kubectl describe po purple | grep -i status
Status:           Pending
  Type                        Status
$ kubectl get po red -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  initContainers:
    - image: busybox
      name: red-initcontainer
      command:
        - sleep
        - "20"
...
$ kubectl replace -f tmp.yaml --force
pod "red" deleted
pod/red replaced
$ kubectl get po orange -oyaml | grep init -A10 | grep comm -A10
  - command:
    - sh
    - -c
    - sleeeep 2;
    image: busybox
    imagePullPolicy: Always
    name: init-myservice
    resources: {}
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
$ kubectl get po orange -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
...
  initContainers:
    - command:
        - sh
        - -c
        - sleep 2
...
$ kubectl replace -f tmp.yaml --force
pod "orange" deleted
pod/orange replaced
```

</details>

## Cluster Maintenance

<details>
<summary>
OS Upgrades
</summary>

```shell
$ kubectl get node
NAME           STATUS   ROLES           AGE     VERSION
controlplane   Ready    control-plane   4m6s    v1.29.0
node01         Ready    <none>          3m19s   v1.29.0
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
blue   3/3     3            3           26s
$ kubectl get po -owide
NAME                    READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
blue-667bf6b9f9-7xnk8   1/1     Running   0          43s   10.244.1.3   node01         <none>           <none>
blue-667bf6b9f9-jkzbz   1/1     Running   0          43s   10.244.0.4   controlplane   <none>           <none>
blue-667bf6b9f9-mfxtv   1/1     Running   0          43s   10.244.1.2   node01         <none>           <none>
$ kubectl drain node01 --ignore-daemonsets
node/node01 cordoned
Warning: ignoring DaemonSet-managed Pods: kube-flannel/kube-flannel-ds-jcdp7, kube-system/kube-proxy-wxt8z
evicting pod default/blue-667bf6b9f9-mfxtv
evicting pod default/blue-667bf6b9f9-7xnk8
pod/blue-667bf6b9f9-mfxtv evicted
pod/blue-667bf6b9f9-7xnk8 evicted
node/node01 drained
$ kubectl get po -owide
NAME                    READY   STATUS    RESTARTS   AGE     IP           NODE           NOMINATED NODE   READINESS GATES
blue-667bf6b9f9-2h8tq   1/1     Running   0          26s     10.244.0.6   controlplane   <none>           <none>
blue-667bf6b9f9-jkzbz   1/1     Running   0          2m44s   10.244.0.4   controlplane   <none>           <none>
blue-667bf6b9f9-k9j9j   1/1     Running   0          26s     10.244.0.5   controlplane   <none>           <none>
$ kubectl uncordon node01
node/node01 uncordoned
$ kubectl drain node01 --ignore-daemonsets
node/node01 cordoned
error: unable to drain node "node01" due to error:cannot delete Pods declare no controller (use --force to override): default/hr-app, continuing command...
There are pending nodes to be drained:
 node01
cannot delete Pods declare no controller (use --force to override): default/hr-app
$ kubectl get po -owide
NAME                    READY   STATUS    RESTARTS   AGE     IP           NODE           NOMINATED NODE   READINESS GATES
blue-667bf6b9f9-2h8tq   1/1     Running   0          4m6s    10.244.0.6   controlplane   <none>           <none>
blue-667bf6b9f9-jkzbz   1/1     Running   0          6m24s   10.244.0.4   controlplane   <none>           <none>
blue-667bf6b9f9-k9j9j   1/1     Running   0          4m6s    10.244.0.5   controlplane   <none>           <none>
hr-app                  1/1     Running   0          114s    10.244.1.4   node01         <none>           <none>
$ kubectl cordon node01
node/node01 cordoned
```

</details>

<details>
<summary>
Cluster Upgrade Process
</summary>

```shell
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   55m   v1.28.0
node01         Ready    <none>          54m   v1.28.0
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
blue   5/5     5            5           52s
$ kubectl get po -owide
NAME                    READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
blue-667bf6b9f9-5s9g8   1/1     Running   0          81s   10.244.0.4   controlplane   <none>           <none>
blue-667bf6b9f9-bfsxp   1/1     Running   0          81s   10.244.1.3   node01         <none>           <none>
blue-667bf6b9f9-jx2dn   1/1     Running   0          81s   10.244.0.5   controlplane   <none>           <none>
blue-667bf6b9f9-ql6fq   1/1     Running   0          81s   10.244.1.4   node01         <none>           <none>
blue-667bf6b9f9-xzl7j   1/1     Running   0          81s   10.244.1.2   node01         <none>           <none>
$ kubeadm upgrade plan
[upgrade/config] Making sure the configuration is correct:
[upgrade/config] Reading configuration from the cluster...
[upgrade/config] FYI: You can look at this config file with 'kubectl -n kube-system get cm kubeadm-config -o yaml'
[preflight] Running pre-flight checks.
[upgrade] Running cluster health checks
[upgrade] Fetching available versions to upgrade to
[upgrade/versions] Cluster version: v1.28.0
[upgrade/versions] kubeadm version: v1.28.0
I0424 08:04:34.188549   15941 version.go:256] remote version is much newer: v1.30.0; falling back to: stable-1.28
[upgrade/versions] Target version: v1.28.9
[upgrade/versions] Latest version in the v1.28 series: v1.28.9

Components that must be upgraded manually after you have upgraded the control plane with 'kubeadm upgrade apply':
COMPONENT   CURRENT       TARGET
kubelet     2 x v1.28.0   v1.28.9

Upgrade to the latest version in the v1.28 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.28.0   v1.28.9
kube-controller-manager   v1.28.0   v1.28.9
kube-scheduler            v1.28.0   v1.28.9
kube-proxy                v1.28.0   v1.28.9
CoreDNS                   v1.10.1   v1.10.1
etcd                      3.5.9-0   3.5.9-0

You can now apply the upgrade by executing the following command:

        kubeadm upgrade apply v1.28.9

Note: Before you can perform this upgrade, you have to update kubeadm to v1.28.9.

_____________________________________________________________________


The table below shows the current state of component configs as understood by this version of kubeadm.
Configs that have a "yes" mark in the "MANUAL UPGRADE REQUIRED" column require manual config upgrade or
resetting to kubeadm defaults before a successful upgrade can be performed. The version to manually
upgrade to is denoted in the "PREFERRED VERSION" column.

API GROUP                 CURRENT VERSION   PREFERRED VERSION   MANUAL UPGRADE REQUIRED
kubeproxy.config.k8s.io   v1alpha1          v1alpha1            no
kubelet.config.k8s.io     v1beta1           v1beta1             no
_____________________________________________________________________
```

```shell Upgrade (Control Plane)
$ echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /
$ apt update
$ apt-cache madison kubeadm
   kubeadm | 1.29.4-2.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt-mark unhold kubeadm
$ apt-get update
$ apt-get install -y kubeadm=1.29.0-1.1
$ apt-mark hold kubeadm
$ kubeadm upgrade apply v1.29.0
...
[upgrade/successful] SUCCESS! Your cluster was upgraded to "v1.29.0". Enjoy!
...
$ kubectl drain controlplane --ignore-daemonsets
node/controlplane cordoned
Warning: ignoring DaemonSet-managed Pods: kube-flannel/kube-flannel-ds-fwjlg, kube-system/kube-proxy-g6hct
evicting pod kube-system/coredns-5dd5756b68-hv2qj
evicting pod default/blue-667bf6b9f9-d87ww
evicting pod default/blue-667bf6b9f9-qxslb
pod/blue-667bf6b9f9-d87ww evicted
pod/blue-667bf6b9f9-qxslb evicted
pod/coredns-5dd5756b68-hv2qj evicted
node/controlplane drained
$ apt-mark unhold kubelet kubectl
$ apt-get update
$ apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
$ apt-mark hold kubelet kubectl
$ systemctl daemon-reload
$ systemctl restart kubelet
$ kubectl uncordon controlplane
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   56m   v1.29.0
node01         Ready    <none>          56m   v1.28.0
```

```shell Upgrade (Worker Node)
$ ssh node01
$ echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /
$ apt update
$ apt-cache madison kubeadm
   kubeadm | 1.29.4-2.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt-mark unhold kubeadm
$ apt-get update
$ apt-get install -y kubeadm=1.29.0-1.1
$ apt-mark hold kubeadm
$ kubeadm upgrade node
[upgrade] Reading configuration from the cluster...
[upgrade] FYI: You can look at this config file with 'kubectl -n kube-system get cm kubeadm-config -o yaml'
[preflight] Running pre-flight checks
[preflight] Skipping prepull. Not a control plane node.
[upgrade] Skipping phase. Not a control plane node.
[upgrade] Backing up kubelet config file to /etc/kubernetes/tmp/kubeadm-kubelet-config3882026834/config.yaml
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[upgrade] The configuration for this node was successfully updated!
[upgrade] Now you should go ahead and upgrade the kubelet package using your package manager.
$ kubectl drain node01 --ignore-daemonsets
node/node01 cordoned
Warning: ignoring DaemonSet-managed Pods: kube-flannel/kube-flannel-ds-ckbls, kube-system/kube-proxy-xjk9s
evicting pod kube-system/coredns-76f75df574-t5tzt
evicting pod default/blue-667bf6b9f9-s4784
evicting pod default/blue-667bf6b9f9-fwjvr
evicting pod default/blue-667bf6b9f9-z9wdm
evicting pod kube-system/coredns-76f75df574-8j4bn
evicting pod default/blue-667bf6b9f9-gcfbj
evicting pod default/blue-667bf6b9f9-z5d9w
pod/blue-667bf6b9f9-fwjvr evicted
pod/blue-667bf6b9f9-z9wdm evicted
pod/blue-667bf6b9f9-gcfbj evicted
pod/blue-667bf6b9f9-s4784 evicted
pod/blue-667bf6b9f9-z5d9w evicted
pod/coredns-76f75df574-t5tzt evicted
pod/coredns-76f75df574-8j4bn evicted
node/node01 drained
$ apt-mark unhold kubelet kubectl
$ apt-get update
$ apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
$ apt-mark hold kubelet kubectl
$ systemctl daemon-reload
$ systemctl restart kubelet
$ kubectl uncordon node01
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   63m   v1.29.0
node01         Ready    <none>          62m   v1.29.0
```

</details>

<details>
<summary>
Backup and Restore Methods (Stacked etcd)
</summary>

```shell
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
blue   3/3     3            3           89s
red    2/2     2            2           89s
$ kubectl describe -n kube-system po etcd-controlplane | grep Image:
    Image:         registry.k8s.io/etcd:3.5.10-0
$ kubectl describe -n kube-system po etcd-controlplane | grep client-url
...
      --listen-client-urls=https://127.0.0.1:2379,https://192.24.189.3:2379
$ kubectl describe -n kube-system po etcd-controlplane | grep cert
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
...
$ kubectl describe -n kube-system po etcd-controlplane | grep ca
...
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
```

```shell Backup etcd
$ kubectl describe -n kube-system po etcd-controlplane | grep ca
Priority Class Name:  system-node-critical
      --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
$ ca=/etc/kubernetes/pki/etcd/ca.crt
$ kubectl describe -n kube-system po etcd-controlplane | grep cert
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
      --client-cert-auth=true
      --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
      --peer-client-cert-auth=true
      /etc/kubernetes/pki/etcd from etcd-certs (rw)
  etcd-certs:
$ cert=/etc/kubernetes/pki/etcd/server.crt
$ kubectl describe -n kube-system po etcd-controlplane | grep key
      --key-file=/etc/kubernetes/pki/etcd/server.key
      --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
$ key=/etc/kubernetes/pki/etcd/server.key
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 --cacert=$ca --cert=$cert --key=$key snapshot save /opt/snapshot-pre-boot.db
Snapshot saved at /opt/snapshot-pre-boot.db
```

```shell Restore etcd
$ ETCDCTL_API=3 etcdctl snapshot restore /opt/snapshot-pre-boot.db --data-dir=/var/lib/etcd-from-backup
2024-04-24 13:05:20.731277 I | mvcc: restore compact to 1301
2024-04-24 13:05:20.737597 I | etcdserver/membership: added member 8e9e05c52164694d [http://localhost:2380] to cluster cdf818194e3a8c32
$ vi /etc/kubernetes/manifests/etcd.yaml
...
  - hostPath:
      path: /var/lib/etcd-from-backup
      type: DirectoryOrCreate
    name: etcd-data
...
$ crictl ps | grep etcd
4d1dabff80b0c       a0eed15eed449       About a minute ago   Running             etcd                      0                   b002687d9f86e       etcd-controlplane
```

</details>

<details>
<summary>
Backup and Restore Methods (External etcd)
</summary>

```shell
$ kubectl config view
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: DATA+OMITTED
    server: https://cluster1-controlplane:6443
  name: cluster1
- cluster:
    certificate-authority-data: DATA+OMITTED
    server: https://192.24.29.9:6443
  name: cluster2
contexts:
- context:
    cluster: cluster1
    user: cluster1
  name: cluster1
- context:
    cluster: cluster2
    user: cluster2
  name: cluster2
current-context: cluster1
kind: Config
preferences: {}
users:
- name: cluster1
  user:
    client-certificate-data: REDACTED
    client-key-data: REDACTED
- name: cluster2
  user:
    client-certificate-data: REDACTED
    client-key-data: REDACTED
$ kubectl config use-context cluster1
Switched to context "cluster1".
$ kubectl get node
NAME                    STATUS   ROLES           AGE   VERSION
cluster1-controlplane   Ready    control-plane   55m   v1.24.0
cluster1-node01         Ready    <none>          54m   v1.24.0
$ kubectl config use-context cluster2
Switched to context "cluster2".
$ kubectl get node
NAME                    STATUS   ROLES           AGE   VERSION
cluster2-controlplane   Ready    control-plane   56m   v1.24.0
cluster2-node01         Ready    <none>          55m   v1.24.0
$ kubectl config use-context cluster1
Switched to context "cluster1".
$ kubectl get -n kube-system po | grep etcd
etcd-cluster1-controlplane                      1/1     Running   0             59m
$ kubectl config use-context cluster2
Switched to context "cluster2".
$ kubectl get -n kube-system po
NAME                                            READY   STATUS    RESTARTS      AGE
coredns-6d4b75cb6d-8gz4r                        1/1     Running   0             60m
coredns-6d4b75cb6d-j5h75                        1/1     Running   0             60m
kube-apiserver-cluster2-controlplane            1/1     Running   0             61m
kube-controller-manager-cluster2-controlplane   1/1     Running   0             61m
kube-proxy-bjkgs                                1/1     Running   0             60m
kube-proxy-nwdpm                                1/1     Running   0             60m
kube-scheduler-cluster2-controlplane            1/1     Running   0             61m
weave-net-mw8ns                                 2/2     Running   1 (60m ago)   60m
weave-net-wgzsg                                 2/2     Running   0             60m
$ ssh cluster2-controlplane
$ ps -ef | grep etcd
root        1789    1381  0 12:27 ?        00:04:07 kube-apiserver --advertise-address=192.24.29.9 --allow-privileged=true --authorization-mode=Node,RBAC --client-ca-file=/etc/kubernetes/pki/ca.crt --enable-admission-plugins=NodeRestriction --enable-bootstrap-token-auth=true --etcd-cafile=/etc/kubernetes/pki/etcd/ca.pem --etcd-certfile=/etc/kubernetes/pki/etcd/etcd.pem --etcd-keyfile=/etc/kubernetes/pki/etcd/etcd-key.pem --etcd-servers=https://192.24.29.21:2379 --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key --requestheader-allowed-names=front-proxy-client --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt --requestheader-extra-headers-prefix=X-Remote-Extra- --requestheader-group-headers=X-Remote-Group --requestheader-username-headers=X-Remote-User --secure-port=6443 --service-account-issuer=https://kubernetes.default.svc.cluster.local --service-account-key-file=/etc/kubernetes/pki/sa.pub --service-account-signing-key-file=/etc/kubernetes/pki/sa.key --service-cluster-ip-range=10.96.0.0/12 --tls-cert-file=/etc/kubernetes/pki/apiserver.crt --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
root        8961    8488  0 13:31 pts/0    00:00:00 grep etcd
$ kubectl -n kube-system describe pod kube-apiserver-cluster2-controlplane | grep etcd
      --etcd-cafile=/etc/kubernetes/pki/etcd/ca.pem
      --etcd-certfile=/etc/kubernetes/pki/etcd/etcd.pem
      --etcd-keyfile=/etc/kubernetes/pki/etcd/etcd-key.pem
      --etcd-servers=https://192.24.29.21:2379
$ kubectl config use-context cluster1
Switched to context "cluster1".
$ kubectl describe -n kube-system po etcd-cluster1-controlplane | grep data
      --data-dir=/var/lib/etcd
...
$ ssh etcd-server
$ ps -ef | grep etcd
etcd         820       1  0 12:27 ?        00:01:18 /usr/local/bin/etcd --name etcd-server --data-dir=/var/lib/etcd-data --cert-file=/etc/etcd/pki/etcd.pem --key-file=/etc/etcd/pki/etcd-key.pem --peer-cert-file=/etc/etcd/pki/etcd.pem --peer-key-file=/etc/etcd/pki/etcd-key.pem --trusted-ca-file=/etc/etcd/pki/ca.pem --peer-trusted-ca-file=/etc/etcd/pki/ca.pem --peer-client-cert-auth --client-cert-auth --initial-advertise-peer-urls https://192.24.29.21:2380 --listen-peer-urls https://192.24.29.21:2380 --advertise-client-urls https://192.24.29.21:2379 --listen-client-urls https://192.24.29.21:2379,https://127.0.0.1:2379 --initial-cluster-token etcd-cluster-1 --initial-cluster etcd-server=https://192.24.29.21:2380 --initial-cluster-state new
root        1048     971  0 14:01 pts/0    00:00:00 grep etcd
$ kubectl describe -n kube-system po etcd-cluster1-controlplane | grep ca
...
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
$ kubectl describe -n kube-system po etcd-cluster1-controlplane | grep cert
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
...
$ kubectl describe -n kube-system po etcd-cluster1-controlplane | grep key
      --key-file=/etc/kubernetes/pki/etcd/server.key
...
$ ssh cluster1-controlplane
$ ca=/etc/kubernetes/pki/etcd/ca.crt
$ cert=/etc/kubernetes/pki/etcd/server.crt
$ key=/etc/kubernetes/pki/etcd/server.key
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 --cacert=$ca --cert=$cert --key=$key snapshot save cluster1.db
Snapshot saved at cluster1.db
$ scp cluster1-controlplane:~/cluster1.db /opt/cluster1.db
cluster1.db                                                                        100% 2092KB 145.0MB/s   00:00
```

```shell Check Member of etcd
$ ca=/etc/etcd/pki/ca.pem
$ cert=/etc/etcd/pki/etcd.pem
$ key=/etc/etcd/pki/etcd-key.pem
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 --cacert=$ca --cert=$cert --key=$key member list
716d168be75b09b8, started, etcd-server, https://192.24.29.21:2380, https://192.24.29.21:2379, false
```

```shell Backup etcd
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 --cacert=$ca --cert=$cert --key=$key snapshot save cluster2.db
{"level":"info","ts":1713968328.701945,"caller":"snapshot/v3_snapshot.go:119","msg":"created temporary db file","path":"cluster2.db.part"}
{"level":"info","ts":"2024-04-24T14:18:48.709Z","caller":"clientv3/maintenance.go:200","msg":"opened snapshot stream; downloading"}
{"level":"info","ts":1713968328.7097495,"caller":"snapshot/v3_snapshot.go:127","msg":"fetching snapshot","endpoint":"https://127.0.0.1:2379"}
{"level":"info","ts":"2024-04-24T14:18:48.726Z","caller":"clientv3/maintenance.go:208","msg":"completed snapshot read; closing"}
{"level":"info","ts":1713968328.7325995,"caller":"snapshot/v3_snapshot.go:142","msg":"fetched snapshot","endpoint":"https://127.0.0.1:2379","size":"2.1 MB","took":0.030502843}
{"level":"info","ts":1713968328.7327907,"caller":"snapshot/v3_snapshot.go:152","msg":"saved","path":"cluster2.db"}
Snapshot saved at cluster2.db
```

```shell Restore etcd
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 --cacert=$ca --cert=$cert --key=$key snapshot restore cluster2.db --data-dir=/var/lib/etcd-data-new
{"level":"info","ts":1713968366.7065794,"caller":"snapshot/v3_snapshot.go:296","msg":"restoring snapshot","path":"cluster2.db","wal-dir":"/var/lib/etcd-data-new/member/wal","data-dir":"/var/lib/etcd-data-new","snap-dir":"/var/lib/etcd-data-new/member/snap"}
{"level":"info","ts":1713968366.7212343,"caller":"mvcc/kvstore.go:388","msg":"restored last compact revision","meta-bucket-name":"meta","meta-bucket-name-key":"finishedCompactRev","restored-compact-revision":7921}
{"level":"info","ts":1713968366.726475,"caller":"membership/cluster.go:392","msg":"added member","cluster-id":"cdf818194e3a8c32","local-member-id":"0","added-peer-id":"8e9e05c52164694d","added-peer-peer-urls":["http://localhost:2380"]}
{"level":"info","ts":1713968366.803066,"caller":"snapshot/v3_snapshot.go:309","msg":"restored snapshot","path":"cluster2.db","wal-dir":"/var/lib/etcd-data-new/member/wal","data-dir":"/var/lib/etcd-data-new","snap-dir":"/var/lib/etcd-data-new/member/snap"}
$ vi /etc/systemd/system/etcd.service
...
  --data-dir=/var/lib/etcd-data-new \
...
$ chown -R etcd:etcd /var/lib/etcd-data-new
$ systemctl daemon-reload
$ systemctl restart etcd
```

</details>

## Security

<details>
<summary>
View Certificate Details
</summary>

```shell
$ kubectl describe -n kube-system po kube-apiserver-controlplane | grep cert | grep tls
      --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
$ kubectl describe -n kube-system po kube-apiserver-controlplane | grep cert | grep etcd
      --etcd-certfile=/etc/kubernetes/pki/apiserver-etcd-client.crt
$ kubectl describe -n kube-system po kube-apiserver-controlplane | grep key | grep kubelet
      --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
$ kubectl describe -n kube-system po etcd-controlplane | grep cert
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
...
$ kubectl describe -n kube-system po etcd-controlplane | grep ca
...
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
$ openssl x509 -in /etc/kubernetes/pki/apiserver.crt -text -noout | grep CN
        Issuer: CN = kubernetes
        Subject: CN = kube-apiserver
$ openssl x509 -in /etc/kubernetes/pki/apiserver.crt -text -noout | grep -i name -A1
            X509v3 Subject Alternative Name:
                DNS:controlplane, DNS:kubernetes, DNS:kubernetes.default, DNS:kubernetes.default.svc, DNS:kubernetes.default.svc.cluster.local, IP Address:10.96.0.1, IP Address:192.21.58.6
$ openssl x509 -in /etc/kubernetes/pki/etcd/server.crt -text -noout | grep CN
        Issuer: CN = etcd-ca
        Subject: CN = controlplane
$ openssl x509 -in /etc/kubernetes/pki/apiserver.crt -text -noout | grep Validity -A2
        Validity
            Not Before: Apr 25 11:12:55 2024 GMT
            Not After : Apr 25 11:17:55 2025 GMT
$ openssl x509 -in /etc/kubernetes/pki/ca.crt -text -noout | grep Validity -A2
        Validity
            Not Before: Apr 25 11:12:55 2024 GMT
            Not After : Apr 23 11:17:55 2034 GMT
$ ls /etc/kubernetes/pki/etcd
ca.crt  ca.key  healthcheck-client.crt  healthcheck-client.key  peer.crt  peer.key  server.crt  server.key
$ vi /etc/kubernetes/manifests/etcd.yaml
...
spec:
  containers:
    - command:
...
        - --cert-file=/etc/kubernetes/pki/etcd/server.crt
...
$ crictl ps -a | grep api
0f1bfe4ccf283       1443a367b16d3       2 seconds ago        Running             kube-apiserver            0                   82f02e827efbf       kube-apiserver-controlplane
$ crictl logs 0f1bfe4ccf283 | head -n1
...
W0425 11:37:18.182665       1 logging.go:59] [core] [Channel #2 SubChannel #3] grpc: addrConn.createTransport failed to connect to {Addr: "127.0.0.1:2379", ServerName: "127.0.0.1:2379", }. Err: connection error: desc = "transport: authentication handshake failed: tls: failed to verify certificate: x509: certificate signed by unknown authority"
F0425 11:37:20.873221       1 instance.go:290] Error creating leases: error creating storage factory: context deadline exceeded
$ vi /etc/kubernetes/manifests/kube-apiserver.yaml
spec:
  containers:
    - command:
...
        - --etcd-cafile=/etc/kubernetes/pki/etcd/ca.crt
```

</details>

<details>
<summary>
Certificates API
</summary>

```shell
$ cat akshay.csr
-----BEGIN CERTIFICATE REQUEST-----
MIICVjCCAT4CAQAwETEPMA0GA1UEAwwGYWtzaGF5MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEA3KPb5FwdhiMhzFFihh/pUT8xWRmhZTLiuLThTwFKnp+i
nF1ywJjViCi14mYpi+ixSu4Akjcttpk1u4q7b8/89gX5747AesHTf0DFSqNY3fm0
Z35xKf6YRrENBHxVMBZJM7RTvOmfdAPLCU4j3imnipbFCQfRNgwWFuUc2eZRdLHL
W3Qizww++OsxUv1fE9Cl5HytjL6buYPZiWpEagL2K0yoAwNq55LMRnvucoN+UU+M
Fiho8Hk13cMe9VM0apTjO5SNIjQ9EQjX84fRhMZL+vclHJLaHNP34KdaIz+ayltZ
oMyPoNc1viXbjwDOM8hs1klf13xnibeRn6u8AFJ0tQIDAQABoAAwDQYJKoZIhvcN
AQELBQADggEBALcbUNSsWLkIlrO1yqpON3TDkyu+FEI0VD9otlteajTciOkT8kAY
RvPdBt7X9vR/Jcbr92qZbj/qVlHBj0sxkn02TR2JUSIQ+NdJ2PoN5uGc5Mve/ie8
GE8i4Aj+TW0HSKRZqQ7HuxTlezOoFxxF5K7HXU8FZfvTe3Z8gzfOmAxXLRysfAqN
RAHmPLi25VdW0epTKQkaBvf2ERciY3+q35SM5Q/AahPqAkq0YrtDbD25oruR4/M+
/XkD67w317qgK7ZgEiwyKkMoRsa9j+wqzva06gnVbErcoY8phfLfQF90jKlAiZd3
1/AngEQJkSH1onyzlGgqZc8Y5Oa4M+ruXQ8=
-----END CERTIFICATE REQUEST-----
$ cat akshay.csr | base64 -w0
LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQ1ZqQ0NBVDRDQVFBd0VURVBNQTBHQTFVRUF3d0dZV3R6YUdGNU1JSUJJakFOQmdrcWhraUc5dzBCQVFFRgpBQU9DQVE4QU1JSUJDZ0tDQVFFQTNLUGI1RndkaGlNaHpGRmloaC9wVVQ4eFdSbWhaVExpdUxUaFR3RktucCtpCm5GMXl3SmpWaUNpMTRtWXBpK2l4U3U0QWtqY3R0cGsxdTRxN2I4Lzg5Z1g1NzQ3QWVzSFRmMERGU3FOWTNmbTAKWjM1eEtmNllSckVOQkh4Vk1CWkpNN1JUdk9tZmRBUExDVTRqM2ltbmlwYkZDUWZSTmd3V0Z1VWMyZVpSZExITApXM1Fpend3KytPc3hVdjFmRTlDbDVIeXRqTDZidVlQWmlXcEVhZ0wySzB5b0F3TnE1NUxNUm52dWNvTitVVStNCkZpaG84SGsxM2NNZTlWTTBhcFRqTzVTTklqUTlFUWpYODRmUmhNWkwrdmNsSEpMYUhOUDM0S2RhSXorYXlsdFoKb015UG9OYzF2aVhiandET004aHMxa2xmMTN4bmliZVJuNnU4QUZKMHRRSURBUUFCb0FBd0RRWUpLb1pJaHZjTgpBUUVMQlFBRGdnRUJBTGNiVU5Tc1dMa0lsck8xeXFwT04zVERreXUrRkVJMFZEOW90bHRlYWpUY2lPa1Q4a0FZClJ2UGRCdDdYOXZSL0pjYnI5MnFaYmovcVZsSEJqMHN4a24wMlRSMkpVU0lRK05kSjJQb041dUdjNU12ZS9pZTgKR0U4aTRBaitUVzBIU0tSWnFRN0h1eFRsZXpPb0Z4eEY1SzdIWFU4RlpmdlRlM1o4Z3pmT21BeFhMUnlzZkFxTgpSQUhtUExpMjVWZFcwZXBUS1FrYUJ2ZjJFUmNpWTMrcTM1U001US9BYWhQcUFrcTBZcnREYkQyNW9ydVI0L00rCi9Ya0Q2N3czMTdxZ0s3WmdFaXd5S2tNb1JzYTlqK3dxenZhMDZnblZiRXJjb1k4cGhmTGZRRjkwaktsQWlaZDMKMS9BbmdFUUprU0gxb255emxHZ3FaYzhZNU9hNE0rcnVYUTg9Ci0tLS0tRU5EIENFUlRJRklDQVRFIFJFUVVFU1QtLS0tLQo=
$ vi tmp.yaml
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: akshay
spec:
  request: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQ1ZqQ0NBVDRDQVFBd0VURVBNQTBHQTFVRUF3d0dZV3R6YUdGNU1JSUJJakFOQmdrcWhraUc5dzBCQVFFRgpBQU9DQVE4QU1JSUJDZ0tDQVFFQTNLUGI1RndkaGlNaHpGRmloaC9wVVQ4eFdSbWhaVExpdUxUaFR3RktucCtpCm5GMXl3SmpWaUNpMTRtWXBpK2l4U3U0QWtqY3R0cGsxdTRxN2I4Lzg5Z1g1NzQ3QWVzSFRmMERGU3FOWTNmbTAKWjM1eEtmNllSckVOQkh4Vk1CWkpNN1JUdk9tZmRBUExDVTRqM2ltbmlwYkZDUWZSTmd3V0Z1VWMyZVpSZExITApXM1Fpend3KytPc3hVdjFmRTlDbDVIeXRqTDZidVlQWmlXcEVhZ0wySzB5b0F3TnE1NUxNUm52dWNvTitVVStNCkZpaG84SGsxM2NNZTlWTTBhcFRqTzVTTklqUTlFUWpYODRmUmhNWkwrdmNsSEpMYUhOUDM0S2RhSXorYXlsdFoKb015UG9OYzF2aVhiandET004aHMxa2xmMTN4bmliZVJuNnU4QUZKMHRRSURBUUFCb0FBd0RRWUpLb1pJaHZjTgpBUUVMQlFBRGdnRUJBTGNiVU5Tc1dMa0lsck8xeXFwT04zVERreXUrRkVJMFZEOW90bHRlYWpUY2lPa1Q4a0FZClJ2UGRCdDdYOXZSL0pjYnI5MnFaYmovcVZsSEJqMHN4a24wMlRSMkpVU0lRK05kSjJQb041dUdjNU12ZS9pZTgKR0U4aTRBaitUVzBIU0tSWnFRN0h1eFRsZXpPb0Z4eEY1SzdIWFU4RlpmdlRlM1o4Z3pmT21BeFhMUnlzZkFxTgpSQUhtUExpMjVWZFcwZXBUS1FrYUJ2ZjJFUmNpWTMrcTM1U001US9BYWhQcUFrcTBZcnREYkQyNW9ydVI0L00rCi9Ya0Q2N3czMTdxZ0s3WmdFaXd5S2tNb1JzYTlqK3dxenZhMDZnblZiRXJjb1k4cGhmTGZRRjkwaktsQWlaZDMKMS9BbmdFUUprU0gxb255emxHZ3FaYzhZNU9hNE0rcnVYUTg9Ci0tLS0tRU5EIENFUlRJRklDQVRFIFJFUVVFU1QtLS0tLQo=
  signerName: kubernetes.io/kube-apiserver-client
  usages:
    - client auth
$ kubectl apply -f tmp.yaml
certificatesigningrequest.certificates.k8s.io/akshay created
$ kubectl get csr
NAME        AGE   SIGNERNAME                                    REQUESTOR                  REQUESTEDDURATION   CONDITION
akshay      30s   kubernetes.io/kube-apiserver-client           kubernetes-admin           <none>              Pending
csr-4v2fw   11m   kubernetes.io/kube-apiserver-client-kubelet   system:node:controlplane   <none>              Approved,Issued
$ kubectl certificate approve akshay
certificatesigningrequest.certificates.k8s.io/akshay approved
$ kubectl get csr
NAME        AGE     SIGNERNAME                                    REQUESTOR                  REQUESTEDDURATION   CONDITION
akshay      2m39s   kubernetes.io/kube-apiserver-client           kubernetes-admin           <none>              Approved,Issued
csr-4v2fw   13m     kubernetes.io/kube-apiserver-client-kubelet   system:node:controlplane   <none>              Approved,Issued
$ kubectl get csr agent-smith -oyaml
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  creationTimestamp: "2024-04-25T11:54:38Z"
  name: agent-smith
  resourceVersion: "1505"
  uid: 959e460f-c4f9-4c3f-9192-db2dcb9d492c
spec:
  groups:
  - system:masters
  - system:authenticated
  request: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQ1dEQ0NBVUFDQVFBd0V6RVJNQThHQTFVRUF3d0libVYzTFhWelpYSXdnZ0VpTUEwR0NTcUdTSWIzRFFFQgpBUVVBQTRJQkR3QXdnZ0VLQW9JQkFRRE8wV0pXK0RYc0FKU0lyanBObzV2UklCcGxuemcrNnhjOStVVndrS2kwCkxmQzI3dCsxZUVuT041TXVxOTlOZXZtTUVPbnJEVU8vdGh5VnFQMncyWE5JRFJYall5RjQwRmJtRCs1eld5Q0sKeTNCaWhoQjkzTUo3T3FsM1VUdlo4VEVMcXlhRGtuUmwvanYvU3hnWGtvazBBQlVUcFdNeDRCcFNpS2IwVSt0RQpJRjVueEF0dE1Wa0RQUTdOYmVaUkc0M2IrUVdsVkdSL3o2RFdPZkpuYmZlek90YUF5ZEdMVFpGQy93VHB6NTJrCkVjQ1hBd3FDaGpCTGt6MkJIUFI0Sjg5RDZYYjhrMzlwdTZqcHluZ1Y2dVAwdEliT3pwcU52MFkwcWRFWnB3bXcKajJxRUwraFpFV2trRno4MGxOTnR5VDVMeE1xRU5EQ25JZ3dDNEdaaVJHYnJBZ01CQUFHZ0FEQU5CZ2txaGtpRwo5dzBCQVFzRkFBT0NBUUVBUzlpUzZDMXV4VHVmNUJCWVNVN1FGUUhVemFsTnhBZFlzYU9SUlFOd0had0hxR2k0CmhPSzRhMnp5TnlpNDRPT2lqeWFENnRVVzhEU3hrcjhCTEs4S2czc3JSRXRKcWw1ckxaeTlMUlZyc0pnaEQ0Z1kKUDlOTCthRFJTeFJPVlNxQmFCMm5XZVlwTTVjSjVURjUzbGVzTlNOTUxRMisrUk1uakRRSjdqdVBFaWM4L2RoawpXcjJFVU02VWF3enlrcmRISW13VHYybWxNWTBSK0ROdFYxWWllKzBIOS9ZRWx0K0ZTR2poNUw1WVV2STFEcWl5CjRsM0UveTNxTDcxV2ZBY3VIM09zVnBVVW5RSVNNZFFzMHFXQ3NiRTU2Q0M1RGhQR1pJcFVibktVcEF3a2ErOEUKdndRMDdqRytocGtueG11RkFlWHhnVXdvZEFMYUo3anUvVERJY3c9PQotLS0tLUVORCBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0K
  signerName: kubernetes.io/kube-apiserver-client
  usages:
  - digital signature
  - key encipherment
  - server auth
  username: agent-x
status: {}
$ kubectl certificate deny agent-smith
certificatesigningrequest.certificates.k8s.io/agent-smith denied
$ kubectl delete csr agent-smith
certificatesigningrequest.certificates.k8s.io "agent-smith" deleted
```

</details>

<details>
<summary>
KubeConfig
</summary>

```shell
$ kubectl config view
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: DATA+OMITTED
    server: https://controlplane:6443
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: kubernetes-admin
  name: kubernetes-admin@kubernetes
current-context: kubernetes-admin@kubernetes
kind: Config
preferences: {}
users:
- name: kubernetes-admin
  user:
    client-certificate-data: DATA+OMITTED
    client-key-data: DATA+OMITTED
$ kubectl config view --kubeconfig my-kube-config
apiVersion: v1
clusters:
- cluster:
    certificate-authority: /etc/kubernetes/pki/ca.crt
    server: https://controlplane:6443
  name: development
- cluster:
    certificate-authority: /etc/kubernetes/pki/ca.crt
    server: https://controlplane:6443
  name: kubernetes-on-aws
...
contexts:
- context:
    cluster: kubernetes-on-aws
    user: aws-user
  name: aws-user@kubernetes-on-aws
- context:
    cluster: test-cluster-1
    user: dev-user
  name: research
...
current-context: test-user@development
kind: Config
preferences: {}
users:
- name: aws-user
  user:
    client-certificate: /etc/kubernetes/pki/users/aws-user/aws-user.crt
    client-key: /etc/kubernetes/pki/users/aws-user/aws-user.key
- name: dev-user
  user:
    client-certificate: /etc/kubernetes/pki/users/dev-user/developer-user.crt
    client-key: /etc/kubernetes/pki/users/dev-user/dev-user.key
- name: test-user
  user:
    client-certificate: /etc/kubernetes/pki/users/test-user/test-user.crt
    client-key: /etc/kubernetes/pki/users/test-user/test-user.key
$ kubectl config --kubeconfig=/root/my-kube-config use-context research
Switched to context "research".
$ kubectl config --kubeconfig=my-kube-config current-context
research
```

</details>

<details>
<summary>
Role Based Access Controls
</summary>

```shell
$ kubectl describe -n kube-system po kube-apiserver-controlplane | grep -i auth
      --authorization-mode=Node,RBAC
...
$ kubectl get role
No resources found in default namespace.
$ kubectl get -A role
NAMESPACE     NAME                                             CREATED AT
blue          developer                                        2024-04-14T11:36:06Z
kube-public   kubeadm:bootstrap-signer-clusterinfo             2024-04-14T11:33:43Z
kube-public   system:controller:bootstrap-signer               2024-04-14T11:33:42Z
kube-system   extension-apiserver-authentication-reader        2024-04-14T11:33:42Z
kube-system   kube-proxy                                       2024-04-14T11:33:43Z
kube-system   kubeadm:kubelet-config                           2024-04-14T11:33:42Z
kube-system   kubeadm:nodes-kubeadm-config                     2024-04-14T11:33:42Z
kube-system   system::leader-locking-kube-controller-manager   2024-04-14T11:33:42Z
kube-system   system::leader-locking-kube-scheduler            2024-04-14T11:33:42Z
kube-system   system:controller:bootstrap-signer               2024-04-14T11:33:42Z
kube-system   system:controller:cloud-provider                 2024-04-14T11:33:42Z
kube-system   system:controller:token-cleaner                  2024-04-14T11:33:42Z
$ kubectl describe -n kube-system role kube-proxy
Name:         kube-proxy
Labels:       <none>
Annotations:  <none>
PolicyRule:
  Resources   Non-Resource URLs  Resource Names  Verbs
  ---------   -----------------  --------------  -----
  configmaps  []                 [kube-proxy]    [get]
$ kubectl describe -n kube-system rolebinding kube-proxy
Name:         kube-proxy
Labels:       <none>
Annotations:  <none>
Role:
  Kind:  Role
  Name:  kube-proxy
Subjects:
  Kind   Name                                             Namespace
  ----   ----                                             ---------
  Group  system:bootstrappers:kubeadm:default-node-token
$ kubectl get po --as dev-user
Error from server (Forbidden): pods is forbidden: User "dev-user" cannot list resource "pods" in API group "" in the namespace "default"
$ kubectl create -n default role developer --verb=list,create,delete --resource=pods
role.rbac.authorization.k8s.io/developer created
$ kubectl create -n default rolebinding dev-user-binding --role=developer --user=dev-user
rolebinding.rbac.authorization.k8s.io/dev-user-binding created
$ kubectl edit -n blue role developer
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  creationTimestamp: "2024-04-14T11:36:06Z"
  name: developer
  namespace: blue
  resourceVersion: "2064"
  uid: 633639f3-1e36-409b-abeb-84ee71a5a63b
rules:
- apiGroups:
  - ""
  resources:
  - pods
  verbs:
  - get
  - watch
  - create
  - delete
- apiGroups:
  - apps
  resources:
  - deployments
  verbs:
  - create
```

</details>

<details>
<summary>
Cluster Roles
</summary>

```shell
$ kubectl get clusterrole | wc -l
72
$ kubectl get clusterrolebinding | wc -l
57
$ kubectl get clusterrolebinding cluster-admin -owide
NAME            ROLE                        AGE   USERS   GROUPS           SERVICEACCOUNTS
cluster-admin   ClusterRole/cluster-admin   25m           system:masters
$ kubectl auth can-i list node --as michelle
Warning: resource 'nodes' is not namespace scoped
no
$ kubectl create clusterrole node-admin --verb=get,watch,list,create,delete --resource=node
clusterrole.rbac.authorization.k8s.io/node-admin created
$ kubectl create clusterrolebinding michelle-binding --clusterrole=node-admin --user=michelle
clusterrolebinding.rbac.authorization.k8s.io/michelle-binding created
$ kubectl auth can-i list node --as michelle
Warning: resource 'nodes' is not namespace scoped
yes
$ kubectl auth can-i list storageclasses --as michelle
Warning: resource 'storageclasses' is not namespace scoped in group 'storage.k8s.io'
no
$ kubectl create clusterrole storage-admin --verb=get,watch,list,create,delete --resource=pv --verb=get,watch,list,create,delete --resource=sc
clusterrole.rbac.authorization.k8s.io/storage-admin created
$ kubectl create clusterrolebinding michelle-storage-admin --clusterrole=storage-admin --user=michelle
clusterrolebinding.rbac.authorization.k8s.io/michelle-storage-admin created
$ kubectl auth can-i list storageclasses --as michelle
Warning: resource 'storageclasses' is not namespace scoped in group 'storage.k8s.io'
yes
```

</details>

<details>
<summary>
Service Accounts
</summary>

```shell
$ kubectl get sa
NAME      SECRETS   AGE
default   0         7m16s
dev       0         12s
$ kubectl describe sa default
Name:                default
Namespace:           default
Labels:              <none>
Annotations:         <none>
Image pull secrets:  <none>
Mountable secrets:   <none>
Tokens:              <none>
Events:              <none>
$ kubectl logs web-dashboard-74cbcd9494-psv2m | grep -i fail
{"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure","message":"pods is forbidden: User \"system:serviceaccount:default:default\" cannot list resource \"pods\" in API group \"\" in the namespace \"default\"","reason":"Forbidden","details":{"kind":"pods"},"code":403}
{"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure","message":"pods is forbidden: User \"system:serviceaccount:default:default\" cannot list resource \"pods\" in API group \"\" in the namespace \"default\"","reason":"Forbidden","details":{"kind":"pods"},"code":403}
$ kubectl describe po web-dashboard-74cbcd9494-psv2m | grep -i service
Service Account:  default
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-k9gh2 (ro)
$ kubectl create sa dashboard-sa
serviceaccount/dashboard-sa created
$ kubectl create token dashboard-sa
eyJhbGciOiJSUzI1NiIsImtpZCI6InJDeGhYaDhLcDJ5ZU92YzdydXpqbmxCelF2TmotUXp2dElBMzZRRmdwYm8ifQ.eyJhdWQiOlsiaHR0cHM6Ly9rdWJlcm5ldGVzLmRlZmF1bHQuc3ZjLmNsdXN0ZXIubG9jYWwiLCJrM3MiXSwiZXhwIjoxNzEzMTA3MDkxLCJpYXQiOjE3MTMxMDM0OTEsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJkZWZhdWx0Iiwic2VydmljZWFjY291bnQiOnsibmFtZSI6ImRhc2hib2FyZC1zYSIsInVpZCI6IjUxNDhhNGVlLTI0MDMtNDQwZS1hYzI1LTczYzgzN2Q1MjExZiJ9fSwibmJmIjoxNzEzMTAzNDkxLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpkYXNoYm9hcmQtc2EifQ.PsWq-vvScMDwCg0Ghp6x0Dr1_UmYVXigEkb4cNAOx2-8EPxSMSGFRJN6DrXM6HzBhObmDppWAIFHHwdCI6UWTReGieJG8Un3J49b132UgppGCOjmrx0cwF5rEZ8a2Y6eHEVsyacXH2E0xDBjybpKLk4KvjXWtSoodkONiPGRA6dIYYpTulpkKF41bgeu30sQ2w3D6qG0cnDl7AkAvypusaMj23eiMHhdUOy5SbTWrn8rSaez18_d_pbsYd_vAA1Sw_ApmhAJ_5b-jmecXmPs5n2Lu-0_E6NRTE5TjBrjnC9fHWUnDEuMAehsQGBRJ27nTF1jJ7CESLi35Wo00Jr7Ig
$ kubectl set sa deploy/web-dashboard dashboard-sa
deployment.apps/web-dashboard serviceaccount updated
```

</details>

<details>
<summary>
Image Security
</summary>

```shell
$ kubectl create secret --help | grep docker
 A docker-registry type secret is for accessing a container registry.
  docker-registry   Create a secret for use with a Docker registry
  kubectl create secret (docker-registry | generic | tls) [options]
$ kubectl create secret docker-registry private-reg-cred --docker-username=dock_user --docker-password=dock_password --docker-server=myprivateregistry.com:5000 --docker-email=dock_user@myprivateregistry.com
secret/private-reg-cred created
$ kubectl edit deploy web
spec:
  template:
    ...
    spec:
      containers:
      - image: myprivateregistry.com:5000/nginx:alpine
        ...
      imagePullSecrets:
      - name: private-reg-cred
...
```

</details>

<details>
<summary>
Security Contexts
</summary>

```shell
$ kubectl get po ubuntu-sleepr -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  ...
  securityContext:
    runAsUser: 1010
...
$ vi tmp.yaml
...
spec:
  containers:
  - ...
    securityContext:
      capabilities:
        add: ["SYS_TIME"]
  ...
  securityContext:
    runAsUser: 0
...
$ vi tmp.yaml
...
spec:
  containers:
  - ...
    securityContext:
      capabilities:
        add: ["SYS_TIME", "NET_ADMIN"]
...
```

</details>

<details>
<summary>
Network Policies
</summary>

```shell
$ kubectl get netpol
NAME             POD-SELECTOR   AGE
payroll-policy   name=payroll   43s
$ kubectl describe netpol payroll-policy
Name:         payroll-policy
Namespace:    default
Created on:   2024-04-15 11:39:36 +0000 UTC
Labels:       <none>
Annotations:  <none>
Spec:
  PodSelector:     name=payroll
  Allowing ingress traffic:
    To Port: 8080/TCP
    From:
      PodSelector: name=internal
  Not affecting egress traffic
  Policy Types: Ingress
$ kubectl get po --show-labels
NAME       READY   STATUS    RESTARTS   AGE     LABELS
external   1/1     Running   0          2m38s   name=external
internal   1/1     Running   0          2m38s   name=internal
mysql      1/1     Running   0          2m38s   name=mysql
payroll    1/1     Running   0          2m38s   name=payroll
$ vi tmp.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: internal-policy
  namespace: default
spec:
  podSelector:
    matchLabels:
      name: internal
  policyTypes:
    - Egress
    - Ingress
  ingress:
    - {}
  egress:
    - to:
        - podSelector:
            matchLabels:
              name: mysql
      ports:
        - protocol: TCP
          port: 3306
    - to:
        - podSelector:
            matchLabels:
              name: payroll
      ports:
        - protocol: TCP
          port: 8080
    - ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP
$ kubectl apply -f tmp.yaml
networkpolicy.networking.k8s.io/internal-policy created
$ kubectl get -n kube-system svc
NAME       TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)                  AGE
kube-dns   ClusterIP   10.96.0.10   <none>        53/UDP,53/TCP,9153/TCP   53m
```

</details>

## Storage

<details>
<summary>
Persistent Volume Claims
</summary>

```shell
$ kubectl exec webapp -- cat /log/app.log
[2024-04-15 11:54:19,239] INFO in event-simulator: USER2 logged out
[2024-04-15 11:54:20,240] INFO in event-simulator: USER4 is viewing page1
...
[2024-04-15 11:54:55,278] INFO in event-simulator: USER1 is viewing page2
[2024-04-15 11:54:56,279] INFO in event-simulator: USER3 is viewing page3
$ kubectl get po webapp -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  containers:
    ...
    - mountPath: /log
      name: log-volume
  ...
  volumes:
  - name: log-volume
    hostPath:
      path: /var/log/webapp
      type: Directory
...
$ kubectl replace -f tmp.yaml --force
pod "webapp" deleted
pod/webapp replaced
$ vi pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-log
spec:
  persistentVolumeReclaimPolicy: Retain
  accessModes:
    - ReadWriteMany
  capacity:
    storage: 100Mi
  hostPath:
    path: /pv/log
$ kubectl apply -f pv.yaml
persistentvolume/pv-log created
$ vi pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: claim-log-1
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Mi
$ kubectl apply -f pvc.yaml
persistentvolumeclaim/claim-log-1 created
$ kubectl get pvc
NAME          STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
claim-log-1   Pending                                                     <unset>                 20s
$ kubectl get pv
NAME     CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      CLAIM   STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-log   100Mi      RWX            Retain           Available                          <unset>                          3m29s
$ kubectl delete pvc claim-log-1
persistentvolumeclaim "claim-log-1" deleted
$ vi pvc.yaml
...
spec:
  accessModes:
    - ReadWriteMany
...
$ kubectl apply -f pvc.yaml
persistentvolumeclaim/claim-log-1 created
$ kubectl get pv
NAME     CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                 STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-log   100Mi      RWX            Retain           Bound    default/claim-log-1                  <unset>                          6m48s
$ kubectl get po -oyaml > tmp.yaml
$ vi tmp.yaml
...
    volumes:
    - name: log-volume
      persistentVolumeClaim:
        claimName: claim-log-1
...
$ kubectl replace -f tmp.yaml --force
pod "webapp" deleted
pod/webapp replaced
$ kubectl delete pvc claim-log-1
persistentvolumeclaim "claim-log-1" deleted
$ kubectl get pvc
NAME          STATUS        VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
claim-log-1   Terminating   pv-log   100Mi      RWX                           <unset>                 5m43s
$ kubectl delete po webapp
pod "webapp" deleted
$ kubectl get pvc
No resources found in default namespace.
$ kubectl get pv
NAME     CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS     CLAIM                 STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-log   100Mi      RWX            Retain           Released   default/claim-log-1                  <unset>                          14m
```

</details>

<details>
<summary>
Storage Class
</summary>

```shell
$ kubectl get sc
NAME                   PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
local-path (default)   rancher.io/local-path   Delete          WaitForFirstConsumer   false                  5m29s
$ kubectl get sc
NAME                        PROVISIONER                     RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
local-path (default)        rancher.io/local-path           Delete          WaitForFirstConsumer   false                  5m49s
local-storage               kubernetes.io/no-provisioner    Delete          WaitForFirstConsumer   false                  6s
portworx-io-priority-high   kubernetes.io/portworx-volume   Delete          Immediate              false                  6s
$ kubectl get pvc
No resources found in default namespace.
$ vi tmp.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: local-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-storage
  resources:
    requests:
      storage: 500Mi
$ kubectl apply -f tmp.yaml
persistentvolumeclaim/local-pvc created
$ kubectl get pvc
NAME        STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS    VOLUMEATTRIBUTESCLASS   AGE
local-pvc   Pending                                      local-storage   <unset>                 30s
$ kubectl describe pvc local-pvc
Name:          local-pvc
Namespace:     default
StorageClass:  local-storage
Status:        Pending
Volume:
Labels:        <none>
Annotations:   <none>
Finalizers:    [kubernetes.io/pvc-protection]
Capacity:
Access Modes:
VolumeMode:    Filesystem
Used By:       <none>
Events:
  Type    Reason                Age               From                         Message
  ----    ------                ----              ----                         -------
  Normal  WaitForFirstConsumer  6s (x7 over 82s)  persistentvolume-controller  waiting for first consumer to be created before binding
$ kubectl run nginx --image nginx:alpine --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
...
      volumeMounts:
        - name: local-pv
          mountPath: /var/www/html
  volumes:
    - name: local-pv
      persistentVolumeClaim:
        claimName: local-pvc
...
$ kubectl get pvc
NAME        STATUS   VOLUME     CAPACITY   ACCESS MODES   STORAGECLASS    VOLUMEATTRIBUTESCLASS   AGE
local-pvc   Bound    local-pv   500Mi      RWO            local-storage   <unset>                 8m12s
$ vi sc.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: delayed-volume-sc
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
$ kubectl apply -f sc.yaml
storageclass.storage.k8s.io/delayed-volume-sc created
```

</details>

## Networking

<details>
<summary>
Explore Environment
</summary>

```shell
$ kubectl get node
NAME           STATUS   ROLES           AGE     VERSION
controlplane   Ready    control-plane   4m15s   v1.29.0
node01         Ready    <none>          3m29s   v1.29.0
$ kubectl describe node controlplane | grep -i ip
...
  InternalIP:  192.27.84.3
$ ip a | grep -B2 192.27.84.3
21304: eth0@if21305: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP group default
    link/ether 02:42:c0:1b:54:03 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 192.27.84.3/24 brd 192.27.84.255 scope global eth0
$ kubectl describe node node01 | grep -i ip
...
  InternalIP:  192.27.84.6
$ ssh node01
$ ip link show eth0
5284: eth0@if5285: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP mode DEFAULT group default
    link/ether 02:42:c0:1b:54:06 brd ff:ff:ff:ff:ff:ff link-netnsid 0
$ ip link show cni0
3: cni0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP mode DEFAULT group default qlen 1000
    link/ether 1a:8a:97:de:5f:9b brd ff:ff:ff:ff:ff:ff
$ ip route show default
default via 172.25.0.1 dev eth1
$ netstat -nplt | grep scheduler
tcp        0      0 127.0.0.1:10259         0.0.0.0:*               LISTEN      3756/kube-scheduler
$ netstat -anp | grep etcd | grep 2380 | wc -l
1
$ netstat -anp | grep etcd | grep 2379 | wc -l
63
```

</details>

<details>
<summary>
CNI
</summary>

```shell
$ ps -aux | grep kubelet | grep --color container-runtime-endpoint
root        4338  0.0  0.0 3925984 99400 ?       Ssl  13:05   0:02 /usr/bin/kubelet --bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf --kubeconfig=/etc/kubernetes/kubelet.conf --config=/var/lib/kubelet/config.yaml --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock --pod-infra-container-image=registry.k8s.io/pause:3.9
$ ls /opt/cni/bin
bandwidth  dhcp   firewall  host-device  ipvlan    macvlan  ptp  static  tuning  vrf
bridge     dummy  flannel   host-local   loopback  portmap  sbr  tap     vlan
$ ls /etc/cni/net.d
10-flannel.conflist
$ cat /etc/cni/net.d/10-flannel.conflist | grep type
      "type": "flannel",
      "type": "portmap",
```

</details>

<details>
<summary>
Depoly Network Solution
</summary>

```shell
$ kubectl describe po | grep -i events -A10
Events:
  Type     Reason                  Age                From               Message
  ----     ------                  ----               ----               -------
  Normal   Scheduled               63s                default-scheduler  Successfully assigned default/app to controlplane
  Warning  FailedCreatePodSandBox  62s                kubelet            Failed to create pod sandbox: rpc error: code = Unknown desc = failed to setup network for sandbox "2ff68c89b91ea666fde3ad2cf3c1a2af1efb4c72d52fda8bde63141b3273a1cb": plugin type="weave-net" name="weave" failed (add): unable to allocate IP address: Post "http://127.0.0.1:6784/ip/2ff68c89b91ea666fde3ad2cf3c1a2af1efb4c72d52fda8bde63141b3273a1cb": dial tcp 127.0.0.1:6784: connect: connection refused
  Normal   SandboxChanged          11s (x5 over 62s)  kubelet            Pod sandbox changed, it will be killed and re-created.
$ kubectl apply -f /root/weave/weave-daemonset-k8s.yaml
serviceaccount/weave-net created
clusterrole.rbac.authorization.k8s.io/weave-net created
clusterrolebinding.rbac.authorization.k8s.io/weave-net created
role.rbac.authorization.k8s.io/weave-net created
rolebinding.rbac.authorization.k8s.io/weave-net created
daemonset.apps/weave-net created
$ kubectl get -A po | grep weave
kube-system   weave-net-tr2q2                        2/2     Running   0          22s
$ kubectl get po
NAME   READY   STATUS    RESTARTS   AGE
app    1/1     Running   0          3m17s
```

</details>

<details>
<summary>
Networking Weave
</summary>

```shell
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   34m   v1.29.0
node01         Ready    <none>          33m   v1.29.0
$ ls /etc/cni/net.d
10-weave.conflist
$ kubectl get -n kube-system po -owide | grep weave
weave-net-4schz                        2/2     Running   1 (36m ago)   36m   192.24.204.6   controlplane   <none>           <none>
weave-net-qxwtm                        2/2     Running   0             35m   192.24.204.9   node01         <none>           <none>
$ ip link | grep weave
4: weave: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1376 qdisc noqueue state UP mode DEFAULT group default qlen 1000
7: vethwe-bridge@vethwe-datapath: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1376 qdisc noqueue master weave state UP mode DEFAULT group default
10: vethwepl0e18de4@if9: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1376 qdisc noqueue master weave state UP mode DEFAULT group default
12: vethwepl5a40e85@if11: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1376 qdisc noqueue master weave state UP mode DEFAULT group default
$ ip a show weave
4: weave: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1376 qdisc noqueue state UP group default qlen 1000
    link/ether a2:53:2c:99:5e:62 brd ff:ff:ff:ff:ff:ff
    inet 10.244.0.1/16 brd 10.244.255.255 scope global weave
       valid_lft forever preferred_lft forever
$ ssh node01
$ ip route | grep weave
10.244.0.0/16 dev weave proto kernel scope link src 10.244.192.0
```

</details>

<details>
<summary>
Service Networking
</summary>

```shell
$ ip a show eth0
7726: eth0@if7727: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP group default
    link/ether 02:42:c0:19:91:09 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 192.25.145.9/24 brd 192.25.145.255 scope global eth0
       valid_lft forever preferred_lft forever
$ ipcalc -b 192.25.145.9
Address:   192.25.145.9
Netmask:   255.255.255.0 = 24
Wildcard:  0.0.0.255
=>
Network:   192.25.145.0/24
HostMin:   192.25.145.1
HostMax:   192.25.145.254
Broadcast: 192.25.145.255
Hosts/Net: 254                   Class C
$ kubectl logs -n kube-system weave-net-c986q | grep range
Defaulted container "weave" out of: weave, weave-npc, weave-init (init)
INFO: 2024/04/15 13:22:07.388133 Command line options: map[conn-limit:200 datapath:datapath db-prefix:/weavedb/weave-net docker-api: expect-npc:true http-addr:127.0.0.1:6784 ipalloc-init:consensus=0 ipalloc-range:10.244.0.0/16 metrics-addr:0.0.0.0:6782 name:1e:19:41:d0:a5:d7 nickname:controlplane no-dns:true no-masq-local:true port:6783]
$ cat /etc/kubernetes/manifests/kube-apiserver.yaml | grep cluster-ip-range
    - --service-cluster-ip-range=10.96.0.0/12
$ kubectl get -n kube-system po | grep kube-proxy
kube-proxy-x5hg8                       1/1     Running   0             52m
kube-proxy-z666n                       1/1     Running   0             52m
$ kubectl logs -n kube-system kube-proxy-x5hg8 | head -n 1
I0415 13:22:03.013123       1 server_others.go:72] "Using iptables proxy"
$ kubectl get -n kube-system ds
NAME         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR            AGE
kube-proxy   2         2         2       2            2           kubernetes.io/os=linux   56m
weave-net    2         2         2       2            2           <none>                   56m
```

</details>

<details>
<summary>
CoreDNS in Kubernetes
</summary>

```shell
$ kubectl get -n kube-system po
NAME                                   READY   STATUS    RESTARTS   AGE
coredns-69f9c977-7xjbk                 1/1     Running   0          103s
coredns-69f9c977-tqb7v                 1/1     Running   0          103s
etcd-controlplane                      1/1     Running   0          117s
kube-apiserver-controlplane            1/1     Running   0          117s
kube-controller-manager-controlplane   1/1     Running   0          117s
kube-proxy-fmxj7                       1/1     Running   0          103s
kube-scheduler-controlplane            1/1     Running   0          2m
$ kubectl get -n kube-system svc
NAME       TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)                  AGE
kube-dns   ClusterIP   10.96.0.10   <none>        53/UDP,53/TCP,9153/TCP   2m55s
$ kubectl get -n kube-system deploy coredns -oyaml | grep -i args | grep etc
      ... "containers":[{"args":["-conf","/etc/coredns/Corefile"], ... "volumeMounts":[{"mountPath":"/etc/coredns" ...
$ kubectl get -n kube-system cm | grep coredns
coredns                                                1      8m18s
$ kubectl describe -n kube-system cm coredns
Name:         coredns
Namespace:    kube-system
Labels:       <none>
Annotations:  <none>

Data
====
Corefile:
----
.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}


BinaryData
====

Events:  <none>
$ kubectl describe svc web-service | grep hr
Selector:          name=hr
$ kubectl edit deploy webapp
...
    spec:
      containers:
      - env:
        - name: DB_Host
          value: mysql.payroll
...
deployment.apps/webapp edited
$ kubectl exec hr -- nslookup mysql.payroll > /root/CKA/nslookup.out
$ cat CKA/nslookup.out
Server:         10.96.0.10
Address:        10.96.0.10#53

Name:   mysql.payroll.svc.cluster.local
Address: 10.96.130.247
```

</details>

<details>
<summary>
Ingress Networking
</summary>

```shell
$ kubectl get -A po | grep ingress
ingress-nginx   ingress-nginx-admission-create-s6bj4        0/1     Completed   0          2m1s
ingress-nginx   ingress-nginx-admission-patch-xdsxn         0/1     Completed   1          2m1s
ingress-nginx   ingress-nginx-controller-7689699d9b-dprv8   1/1     Running     0          2m1s
$ kubectl get -A deploy | grep ingress
ingress-nginx   ingress-nginx-controller   1/1     1            1           2m36s
$ kubectl get -A svc | grep service
app-space       default-backend-service              ClusterIP   10.110.40.158    <none>        80/TCP                       3m5s
app-space       video-service                        ClusterIP   10.104.218.65    <none>        8080/TCP                     3m5s
app-space       wear-service                         ClusterIP   10.104.156.169   <none>        8080/TCP                     3m5s
$ kubectl get -n app-space deploy
NAME              READY   UP-TO-DATE   AVAILABLE   AGE
default-backend   1/1     1            1           3m31s
webapp-video      1/1     1            1           3m31s
webapp-wear       1/1     1            1           3m31s
$ kubectl get -A ing
NAMESPACE   NAME                 CLASS    HOSTS   ADDRESS         PORTS   AGE
app-space   ingress-wear-watch   <none>   *       10.111.45.184   80      3m53s
$ kubectl describe -n app-space ing ingress-wear-watch
Name:             ingress-wear-watch
Labels:           <none>
Namespace:        app-space
Address:          10.111.45.184
Ingress Class:    <none>
Default backend:  <default>
Rules:
  Host        Path  Backends
  ----        ----  --------
  *
              /wear    wear-service:8080 (10.244.0.4:8080)
              /watch   video-service:8080 (10.244.0.5:8080)
Annotations:  nginx.ingress.kubernetes.io/rewrite-target: /
              nginx.ingress.kubernetes.io/ssl-redirect: false
Events:
  Type    Reason  Age                From                      Message
  ----    ------  ----               ----                      -------
  Normal  Sync    10m (x2 over 10m)  nginx-ingress-controller  Scheduled for sync
$ kubectl get -n ingress-nginx deploy ingress-nginx-controller -oyaml | grep default
        - --default-backend-service=app-space/default-backend-service
...
$ kubectl edit -n app-space ing
...
spec:
  rules:
  - http:
      paths:
...
      - backend:
          service:
            name: video-service
            port:
              number: 8080
        path: /stream
        pathType: Prefix
...
ingress.networking.k8s.io/ingress-wear-watch edited
$ kubectl get -n app-space svc
NAME                      TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
default-backend-service   ClusterIP   10.110.40.158    <none>        80/TCP     17m
food-service              ClusterIP   10.100.7.114     <none>        8080/TCP   21s
video-service             ClusterIP   10.104.218.65    <none>        8080/TCP   17m
wear-service              ClusterIP   10.104.156.169   <none>        8080/TCP   17m
$ kubectl edit -n app-space ing
...
spec:
  rules:
  - http:
      paths:
...
      - backend:
          service:
            name: food-service
            port:
              number: 8080
        path: /eat
        pathType: Prefix
...
ingress.networking.k8s.io/ingress-wear-watch edited
$ kubectl get -A po
NAMESPACE        NAME                                        READY   STATUS      RESTARTS   AGE
...
critical-space   webapp-pay-657d677c99-txsbn                 1/1     Running     0          26s
...
$ kubectl get -n critical-space deploy
NAME         READY   UP-TO-DATE   AVAILABLE   AGE
webapp-pay   1/1     1            1           74s
$ kubectl get -n app-space ing -oyaml > tmp.yaml
$ kubectl get -n critical-space svc
NAME          TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)    AGE
pay-service   ClusterIP   10.97.90.60   <none>        8282/TCP   4m56s
$ vi tmp.yaml
...
  spec:
    rules:
    - http:
        paths:
        - backend:
            service:
              name: pay-service
              port:
                number: 8282
          path: /pay
          pathType: Prefix
...
$ kubectl apply -n critical-space -f tmp.yaml
ingress.networking.k8s.io/ingress-wear-watch created
```

```shell
$ kubectl create ns ingress-nginx
namespace/ingress-nginx created
$ kubectl create -n ingress-nginx cm ingress-nginx-controller
configmap/ingress-nginx-controller created
$ kubectl create -n ingress-nginx sa ingress-nginx
serviceaccount/ingress-nginx created
$ kubectl create -n ingress-nginx sa ingress-nginx-admission
serviceaccount/ingress-nginx-admission created
$ cat ingress-controller.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app.kubernetes.io/component: controller
    app.kubernetes.io/instance: ingress-nginx
    app.kubernetes.io/managed-by: Helm
    app.kubernetes.io/name: ingress-nginx
    app.kubernetes.io/part-of: ingress-nginx
    app.kubernetes.io/version: 1.1.2
    helm.sh/chart: ingress-nginx-4.0.18
  name: ingress-nginx-controller
  namespace: ingress-nginx
spec:
  minReadySeconds: 0
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app.kubernetes.io/component: controller
      app.kubernetes.io/instance: ingress-nginx
      app.kubernetes.io/name: ingress-nginx
  template:
    metadata:
      labels:
        app.kubernetes.io/component: controller
        app.kubernetes.io/instance: ingress-nginx
        app.kubernetes.io/name: ingress-nginx
    spec:
      containers:
        - args:
            - /nginx-ingress-controller
            - --publish-service=$(POD_NAMESPACE)/ingress-nginx-controller
            - --election-id=ingress-controller-leader
            - --watch-ingress-without-class=true
            - --default-backend-service=app-space/default-http-backend
            - --controller-class=k8s.io/ingress-nginx
            - --ingress-class=nginx
            - --configmap=$(POD_NAMESPACE)/ingress-nginx-controller
            - --validating-webhook=:8443
            - --validating-webhook-certificate=/usr/local/certificates/cert
            - --validating-webhook-key=/usr/local/certificates/key
          env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: LD_PRELOAD
              value: /usr/local/lib/libmimalloc.so
          image: registry.k8s.io/ingress-nginx/controller:v1.1.2@sha256:28b11ce69e57843de44e3db6413e98d09de0f6688e33d4bd384002a44f78405c
          imagePullPolicy: IfNotPresent
          lifecycle:
            preStop:
              exec:
                command:
                  - /wait-shutdown
          livenessProbe:
            failureThreshold: 5
            httpGet:
              path: /healthz
              port: 10254
              scheme: HTTP
            initialDelaySeconds: 10
            periodSeconds: 10
            successThreshold: 1
            timeoutSeconds: 1
          name: controller
          ports:
            - name: http
              containerPort: 80
              protocol: TCP
            - containerPort: 443
              name: https
              protocol: TCP
            - containerPort: 8443
              name: webhook
              protocol: TCP
          readinessProbe:
            failureThreshold: 3
            httpGet:
              path: /healthz
              port: 10254
              scheme: HTTP
            initialDelaySeconds: 10
            periodSeconds: 10
            successThreshold: 1
            timeoutSeconds: 1
          resources:
            requests:
              cpu: 100m
              memory: 90Mi
          securityContext:
            allowPrivilegeEscalation: true
            capabilities:
              add:
                - NET_BIND_SERVICE
              drop:
                - ALL
            runAsUser: 101
          volumeMounts:
            - mountPath: /usr/local/certificates/
              name: webhook-cert
              readOnly: true
      dnsPolicy: ClusterFirst
      nodeSelector:
        kubernetes.io/os: linux
      serviceAccountName: ingress-nginx
      terminationGracePeriodSeconds: 300
      volumes:
        - name: webhook-cert
          secret:
            secretName: ingress-nginx-admission
---
apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    app.kubernetes.io/component: controller
    app.kubernetes.io/instance: ingress-nginx
    app.kubernetes.io/managed-by: Helm
    app.kubernetes.io/name: ingress-nginx
    app.kubernetes.io/part-of: ingress-nginx
    app.kubernetes.io/version: 1.1.2
    helm.sh/chart: ingress-nginx-4.0.18
  name: ingress-nginx-controller
  namespace: ingress-nginx
spec:
  ports:
    - port: 80
      protocol: TCP
      targetPort: 80
      nodePort: 30080
  selector:
    app.kubernetes.io/component: controller
    app.kubernetes.io/instance: ingress-nginx
    app.kubernetes.io/name: ingress-nginx
  type: NodePort
$ kubectl apply -f ingress-controller.yaml
deployment.apps/ingress-nginx-controller created
service/ingress-nginx-controller created
$ vi tmp.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress-wear-watch
  namespace: app-space
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
    - http:
        paths:
          - path: /wear
            pathType: Prefix
            backend:
              service:
                name: wear-service
                port:
                  number: 8080
          - path: /watch
            pathType: Prefix
            backend:
              service:
                name: video-service
                port:
                  number: 8080
$ kubectl apply -f tmp.yaml
ingress.networking.k8s.io/ingress-wear-watch created
```

</details>

## Install

<details>
<summary>
Cluster Installation using Kubeadm
</summary>

```shell Install Kubeadm
$ cat <<EOF | tee /etc/modules-load.d/k8s.conf
br_netfilter
EOF
$ cat <<EOF | tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF
$ sysctl --system
* Applying /etc/sysctl.d/10-console-messages.conf ...
...
* Applying /etc/sysctl.conf ...
$ apt-get update
$ apt-get install -y apt-transport-https ca-certificates curl
$ mkdir -m 755 /etc/apt/keyrings
$ curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
$ echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list
$ apt-get update
$ apt-cache madison kubeadm
   kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt-get install -y kubelet=1.29.0-1.1 kubeadm=1.29.0-1.1 kubectl=1.29.0-1.1
$ apt-mark hold kubelet kubeadm kubectl
$ ssh node01
$ apt-get update
$ apt-get install -y apt-transport-https ca-certificates curl
$ mkdir -m 755 /etc/apt/keyrings
$ curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
$ echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list
$ apt-get update
$ apt-cache madison kubeadm
   kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt-get install -y kubelet=1.29.0-1.1 kubeadm=1.29.0-1.1 kubectl=1.29.0-1.1
$ apt-mark hold kubelet kubeadm kubectl
```

```shell Init Kubeadm (Control Plane)
$ ifconfig eth0
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1450
        inet 192.32.77.12  netmask 255.255.255.0  broadcast 192.32.77.255
        ether 02:42:c0:20:4d:0c  txqueuelen 0  (Ethernet)
        RX packets 6009  bytes 715278 (715.2 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 5185  bytes 2019890 (2.0 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
$ kubeadm init --apiserver-cert-extra-sans=controlplane --apiserver-advertise-address 192.32.77.12 --pod-network-cidr=10.244.0.0/16
pod-network-cidr=10.244.0.0/16
[init] Using Kubernetes version: v1.29.3
[preflight] Running pre-flight checks
[preflight] Pulling images required for setting up a Kubernetes cluster
[preflight] This might take a minute or two, depending on the speed of your internet connection
[preflight] You can also perform this action in beforehand using 'kubeadm config images pull'
W0416 10:27:50.805148   13986 checks.go:835] detected that the sandbox image "k8s.gcr.io/pause:3.6" of the container runtime is inconsistent with that used by kubeadm. It is recommended that using "registry.k8s.io/pause:3.9" as the CRI sandbox image.
[certs] Using certificateDir folder "/etc/kubernetes/pki"
[certs] Generating "ca" certificate and key
[certs] Generating "apiserver" certificate and key
[certs] apiserver serving cert is signed for DNS names [controlplane kubernetes kubernetes.default kubernetes.default.svc kubernetes.default.svc.cluster.local] and IPs [10.96.0.1 192.32.77.12]
[certs] Generating "apiserver-kubelet-client" certificate and key
[certs] Generating "front-proxy-ca" certificate and key
[certs] Generating "front-proxy-client" certificate and key
[certs] Generating "etcd/ca" certificate and key
[certs] Generating "etcd/server" certificate and key
[certs] etcd/server serving cert is signed for DNS names [controlplane localhost] and IPs [192.32.77.12 127.0.0.1 ::1]
[certs] Generating "etcd/peer" certificate and key
[certs] etcd/peer serving cert is signed for DNS names [controlplane localhost] and IPs [192.32.77.12 127.0.0.1 ::1]
[certs] Generating "etcd/healthcheck-client" certificate and key
[certs] Generating "apiserver-etcd-client" certificate and key
[certs] Generating "sa" key and public key
[kubeconfig] Using kubeconfig folder "/etc/kubernetes"
[kubeconfig] Writing "admin.conf" kubeconfig file
[kubeconfig] Writing "super-admin.conf" kubeconfig file
[kubeconfig] Writing "kubelet.conf" kubeconfig file
[kubeconfig] Writing "controller-manager.conf" kubeconfig file
[kubeconfig] Writing "scheduler.conf" kubeconfig file
[etcd] Creating static Pod manifest for local etcd in "/etc/kubernetes/manifests"
[control-plane] Using manifest folder "/etc/kubernetes/manifests"
[control-plane] Creating static Pod manifest for "kube-apiserver"
[control-plane] Creating static Pod manifest for "kube-controller-manager"
[control-plane] Creating static Pod manifest for "kube-scheduler"
[kubelet-start] Writing kubelet environment file with flags to file "/var/lib/kubelet/kubeadm-flags.env"
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[kubelet-start] Starting the kubelet
[wait-control-plane] Waiting for the kubelet to boot up the control plane as static Pods from directory "/etc/kubernetes/manifests". This can take up to 4m0s
[apiclient] All control plane components are healthy after 12.001326 seconds
[upload-config] Storing the configuration used in ConfigMap "kubeadm-config" in the "kube-system" Namespace
[kubelet] Creating a ConfigMap "kubelet-config" in namespace kube-system with the configuration for the kubelets in the cluster
[upload-certs] Skipping phase. Please see --upload-certs
[mark-control-plane] Marking the node controlplane as control-plane by adding the labels: [node-role.kubernetes.io/control-plane node.kubernetes.io/exclude-from-external-load-balancers]
[mark-control-plane] Marking the node controlplane as control-plane by adding the taints [node-role.kubernetes.io/control-plane:NoSchedule]
[bootstrap-token] Using token: 3r2ibb.b5e2fepi9uzv3o1a
[bootstrap-token] Configuring bootstrap tokens, cluster-info ConfigMap, RBAC Roles
[bootstrap-token] Configured RBAC rules to allow Node Bootstrap tokens to get nodes
[bootstrap-token] Configured RBAC rules to allow Node Bootstrap tokens to post CSRs in order for nodes to get long term certificate credentials
[bootstrap-token] Configured RBAC rules to allow the csrapprover controller automatically approve CSRs from a Node Bootstrap Token
[bootstrap-token] Configured RBAC rules to allow certificate rotation for all node client certificates in the cluster
[bootstrap-token] Creating the "cluster-info" ConfigMap in the "kube-public" namespace
[kubelet-finalize] Updating "/etc/kubernetes/kubelet.conf" to point to a rotatable kubelet client certificate and key
[addons] Applied essential addon: CoreDNS
[addons] Applied essential addon: kube-proxy

Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:

  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

Alternatively, if you are the root user, you can run:

  export KUBECONFIG=/etc/kubernetes/admin.conf

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.32.77.12:6443 --token 3r2ibb.b5e2fepi9uzv3o1a \
        --discovery-token-ca-cert-hash sha256:a69a0b6f38bd3118bcb7f1bc831dcb433b5203a5c4e80ec63a4ff4fe34e968f5
$ mkdir -p $HOME/.kube
$ cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
$ chown $(id -u):$(id -g) $HOME/.kube/config
```

```shell Init Kubeadm (Worker Node)
$ kubeadm token create --print-join-command
kubeadm join 192.32.77.12:6443 --token l70pml.o66rknoiu5bxrp92 --discovery-token-ca-cert-hash sha256:a69a0b6f38bd3118bcb7f1bc831dcb433b5203a5c4e80ec63a4ff4fe34e968f5
$ kubeadm join 192.32.77.12:6443 --token l70pml.o66rknoiu5bxrp92 --discovery-token-ca-cert-hash sha256:a69a0b6f38bd3118bcb7f1bc831dcb433b5203a5c4e80ec63a4ff4fe34e968f5
[preflight] Running pre-flight checks
[preflight] Reading configuration from the cluster...
[preflight] FYI: You can look at this config file with 'kubectl -n kube-system get cm kubeadm-config -o yaml'
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[kubelet-start] Writing kubelet environment file with flags to file "/var/lib/kubelet/kubeadm-flags.env"
[kubelet-start] Starting the kubelet
[kubelet-start] Waiting for the kubelet to perform the TLS Bootstrap...

This node has joined the cluster:
* Certificate signing request was sent to apiserver and a response was received.
* The Kubelet was informed of the new secure connection details.

Run 'kubectl get nodes' on the control-plane to see this node join the cluster.
$ kubectl get node
NAME           STATUS     ROLES           AGE    VERSION
controlplane   NotReady   control-plane   4m1s   v1.29.0
node01         NotReady   <none>          26s    v1.29.0
```

```shell Install CNI
$ curl -LO https://raw.githubusercontent.com/flannel-io/flannel/v0.20.2/Documentation/kube-flannel.yml
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  4591  100  4591    0     0  30606      0 --:--:-- --:--:-- --:--:-- 30812
$ vi kube-flannel.yml
...
spec:
...
  template:
...
    spec:
...
      containers:
      - name: kube-flannel
       #image: flannelcni/flannel:v0.20.2 for ppc64le and mips64le (dockerhub limitations may apply)
        image: docker.io/rancher/mirrored-flannelcni-flannel:v0.20.2
        command:
        - /opt/bin/flanneld
        args:
        - --ip-masq
        - --kube-subnet-mgr
        - --iface=eth0
...
$ kubectl apply -f kube-flannel.yml
namespace/kube-flannel created
clusterrole.rbac.authorization.k8s.io/flannel created
clusterrolebinding.rbac.authorization.k8s.io/flannel created
serviceaccount/flannel created
configmap/kube-flannel-cfg created
daemonset.apps/kube-flannel-ds created
```

</details>

## Troubleshooting

<details>
<summary>
Application Failure
</summary>

```shell Service: Name
k get -n alpha all
NAME                               READY   STATUS    RESTARTS   AGE
pod/webapp-mysql-b68bb6bc8-crtgx   1/1     Running   0          87s
pod/mysql                          1/1     Running   0          88s

NAME                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql         ClusterIP   10.43.236.37    <none>        3306/TCP         88s
service/web-service   NodePort    10.43.193.251   <none>        8080:30081/TCP   87s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           87s

NAME                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-b68bb6bc8   1         1         1       87s
$ kubectl get -n alpha svc mysql -oyaml > tmp.yaml
$ kubectl delete -n alpha svc mysql
$ vi tmp.yaml
...
metadata:
...
  name: mysql-service
...
$ kubectl apply -f tmp.yaml -n alpha
service/mysql-service created
```

```shell Service: Port
$ kubectl get -n beta all
NAME                               READY   STATUS    RESTARTS   AGE
pod/webapp-mysql-b68bb6bc8-wlfgj   1/1     Running   0          31s
pod/mysql                          1/1     Running   0          31s

NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql-service   ClusterIP   10.43.237.70    <none>        3306/TCP         31s
service/web-service     NodePort    10.43.173.146   <none>        8080:30081/TCP   31s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           31s

NAME                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-b68bb6bc8   1         1         1       31s
$ kubectl edit -n beta svc mysql-service
...
  ports:
  - port: 3306
    protocol: TCP
    targetPort: 3306
...
service/mysql-service edited
```

```shell Service: Selector
$ kubectl get -n gamma all
NAME                               READY   STATUS    RESTARTS   AGE
pod/mysql                          1/1     Running   0          19s
pod/webapp-mysql-b68bb6bc8-qn2j5   1/1     Running   0          19s

NAME                    TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)          AGE
service/mysql-service   ClusterIP   10.43.14.61   <none>        3306/TCP         19s
service/web-service     NodePort    10.43.28.69   <none>        8080:30081/TCP   19s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           19s

NAME                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-b68bb6bc8   1         1         1       19s
$ kubectl edit -n gamma svc mysql-service
...
  selector:
    name: mysql
...
service/mysql-service edited
```

```shell Deployment: Environment
$ kubectl get -n delta all
NAME                               READY   STATUS    RESTARTS   AGE
pod/mysql                          1/1     Running   0          33s
pod/webapp-mysql-785cd8f94-bjbjd   1/1     Running   0          33s

NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql-service   ClusterIP   10.43.217.118   <none>        3306/TCP         33s
service/web-service     NodePort    10.43.229.38    <none>        8080:30081/TCP   32s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           33s

NAME                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-785cd8f94   1         1         1       33s
$ kubectl edit -n delta deploy webapp-mysql
...
    spec:
      containers:
      - env:
        - name: DB_Host
          value: mysql-service
        - name: DB_User
          value: root
        - name: DB_Password
          value: paswrd
...
deployment.apps/webapp-mysql edited
```

```shell Pod & Deployment: Environment
$ kubectl get -n epsilon all
NAME                               READY   STATUS    RESTARTS   AGE
pod/mysql                          1/1     Running   0          29s
pod/webapp-mysql-785cd8f94-mvlwb   1/1     Running   0          29s

NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql-service   ClusterIP   10.43.141.236   <none>        3306/TCP         29s
service/web-service     NodePort    10.43.34.96     <none>        8080:30081/TCP   29s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           29s

NAME                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-785cd8f94   1         1         1       29s
$ kubectl get -n epsilon po mysql -oyaml > tmp.yaml
$ vi tmp.yaml
...
spec:
  containers:
  - env:
    - name: MYSQL_ROOT_PASSWORD
      value: paswrd
...
$ kubectl replace -f tmp.yaml -n epsilon --force
pod "mysql" deleted
pod/mysql replaced
$ kubectl edit -n epsilon deploy webapp-mysql
...
    spec:
      containers:
      - env:
        - name: DB_Host
          value: mysql-service
        - name: DB_User
          value: root
        - name: DB_Password
          value: paswrd
...
deployment.apps/webapp-mysql edited
```

```shell Service: Port
$ kubectl get -n zeta all
NAME                               READY   STATUS    RESTARTS   AGE
pod/mysql                          1/1     Running   0          33s
pod/webapp-mysql-785cd8f94-tg9tb   1/1     Running   0          33s

NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql-service   ClusterIP   10.43.152.145   <none>        3306/TCP         33s
service/web-service     NodePort    10.43.137.225   <none>        8080:30088/TCP   33s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           33s

NAME                                     DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-785cd8f94   1         1         1       33s
$ kubectl edit -n zeta svc web-service
...
  ports:
  - nodePort: 30081
    port: 8080
    protocol: TCP
    targetPort: 8080
...
service/web-service edited
$ kubectl edit -n zeta deploy webapp-mysql
deployment.apps/webapp-mysql edited
$ kubectl get -n zeta po mysql -oyaml > tmp.yaml
$ vi tmp.yaml
$ kubectl replace -f tmp.yaml -n zeta --force
pod "mysql" deleted
pod/mysql replaced
```

</details>

<details>
<summary>
Control Plane Failure
</summary>

```shell kube-scheduler: Command
$ kubectl get -n kube-system po
NAME                                   READY   STATUS             RESTARTS      AGE
coredns-69f9c977-jmwn8                 1/1     Running            0             6m59s
coredns-69f9c977-qh7ps                 1/1     Running            0             6m59s
etcd-controlplane                      1/1     Running            0             7m12s
kube-apiserver-controlplane            1/1     Running            0             7m12s
kube-controller-manager-controlplane   1/1     Running            0             7m12s
kube-proxy-8fprn                       1/1     Running            0             6m59s
kube-scheduler-controlplane            0/1     CrashLoopBackOff   5 (58s ago)   4m19s
$ kubectl describe -n kube-system po kube-scheduler-controlplane | grep -i events -A10
Events:
  Type     Reason   Age                    From     Message
  ----     ------   ----                   ----     -------
  Normal   Pulled   5m43s (x4 over 6m39s)  kubelet  Container image "registry.k8s.io/kube-scheduler:v1.29.0" already present on machine
  Normal   Created  5m43s (x4 over 6m39s)  kubelet  Created container kube-scheduler
  Warning  Failed   5m42s (x4 over 6m39s)  kubelet  Error: failed to create containerd task: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: exec: "kube-schedulerrrr": executable file not found in $PATH: unknown
  Warning  BackOff  97s (x31 over 6m37s)   kubelet  Back-off restarting failed container kube-scheduler in pod kube-scheduler-controlplane_kube-system(23ace3c8b1dea5b6d6b30e6bcbb810a7)
$ vi /etc/kubernetes/manifests/kube-scheduler.yaml
...
spec:
  containers:
  - command:
    - kube-scheduler
...
$ kubectl get -n kube-system po
NAME                                   READY   STATUS    RESTARTS   AGE
coredns-69f9c977-jmwn8                 1/1     Running   0          11m
coredns-69f9c977-qh7ps                 1/1     Running   0          11m
etcd-controlplane                      1/1     Running   0          11m
kube-apiserver-controlplane            1/1     Running   0          11m
kube-controller-manager-controlplane   1/1     Running   0          11m
kube-proxy-8fprn                       1/1     Running   0          11m
kube-scheduler-controlplane            1/1     Running   0          26s
```

```shell kube-controller-manager: Command
$ kubectl scale deploy app --replicas 2
deployment.apps/app scaled
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
app    1/2     1            1           12m
$ kubectl get -n kube-system po
NAME                                   READY   STATUS             RESTARTS       AGE
coredns-69f9c977-jmwn8                 1/1     Running            0              16m
coredns-69f9c977-qh7ps                 1/1     Running            0              16m
etcd-controlplane                      1/1     Running            0              17m
kube-apiserver-controlplane            1/1     Running            0              17m
kube-controller-manager-controlplane   0/1     CrashLoopBackOff   5 (107s ago)   5m8s
kube-proxy-8fprn                       1/1     Running            0              16m
kube-scheduler-controlplane            1/1     Running            0              6m9s
$ kubectl describe -n kube-system po kube-controller-manager-controlplane | grep -i events -A10
Events:
  Type     Reason   Age                    From     Message
  ----     ------   ----                   ----     -------
  Normal   Started  4m36s (x4 over 5m44s)  kubelet  Started container kube-controller-manager
  Normal   Pulled   3m53s (x5 over 5m44s)  kubelet  Container image "registry.k8s.io/kube-controller-manager:v1.29.0" already present on machine
  Normal   Created  3m53s (x5 over 5m44s)  kubelet  Created container kube-controller-manager
  Warning  BackOff  40s (x27 over 5m39s)   kubelet  Back-off restarting failed container kube-controller-manager in pod kube-controller-manager-controlplane_kube-system(bdb1c9251d08619760167697af010d5c)
$ vi /etc/kubernetes/manifests/kube-controller-manager.yaml
...
  - command:
...
    - --kubeconfig=/etc/kubernetes/controller-manager.conf
...
$ kubectl get -n kube-system po
NAME                                   READY   STATUS    RESTARTS   AGE
coredns-69f9c977-jmwn8                 1/1     Running   0          19m
coredns-69f9c977-qh7ps                 1/1     Running   0          19m
etcd-controlplane                      1/1     Running   0          20m
kube-apiserver-controlplane            1/1     Running   0          20m
kube-controller-manager-controlplane   1/1     Running   0          26s
kube-proxy-8fprn                       1/1     Running   0          19m
kube-scheduler-controlplane            1/1     Running   0          9m6s
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
app    2/2     2            2           17m
```

```shell kube-controller-manager: hostPath
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
app    2/3     2            2           19m
$ kubectl get -n kube-system po
NAME                                   READY   STATUS             RESTARTS      AGE
coredns-69f9c977-jmwn8                 1/1     Running            0             22m
coredns-69f9c977-qh7ps                 1/1     Running            0             22m
etcd-controlplane                      1/1     Running            0             22m
kube-apiserver-controlplane            1/1     Running            0             22m
kube-controller-manager-controlplane   0/1     CrashLoopBackOff   3 (38s ago)   98s
kube-proxy-8fprn                       1/1     Running            0             22m
kube-scheduler-controlplane            1/1     Running            0             11m
$ kubectl describe -n kube-system po kube-controller-manager-controlplane | grep -i events -A10
Events:
  Type     Reason   Age                   From     Message
  ----     ------   ----                  ----     -------
  Normal   Started  85s (x4 over 2m23s)   kubelet  Started container kube-controller-manager
  Warning  BackOff  49s (x11 over 2m18s)  kubelet  Back-off restarting failed container kube-controller-manager in pod kube-controller-manager-controlplane_kube-system(f3acd5d8bac629a5bb29ac545fabcc86)
  Normal   Pulled   36s (x5 over 2m23s)   kubelet  Container image "registry.k8s.io/kube-controller-manager:v1.29.0" already present on machine
  Normal   Created  35s (x5 over 2m23s)   kubelet  Created container kube-controller-manager
$ vi /etc/kubernetes/manifests/kube-controller-manager.yaml
...
  volumes:
...
  - hostPath:
      path: /etc/kubernetes/pki
...
$ kubectl get -n kube-system po
NAME                                   READY   STATUS    RESTARTS   AGE
coredns-69f9c977-jmwn8                 1/1     Running   0          26m
coredns-69f9c977-qh7ps                 1/1     Running   0          26m
etcd-controlplane                      1/1     Running   0          26m
kube-apiserver-controlplane            1/1     Running   0          26m
kube-controller-manager-controlplane   1/1     Running   0          24s
kube-proxy-8fprn                       1/1     Running   0          26m
kube-scheduler-controlplane            1/1     Running   0          15m
$ kubectl get deploy
NAME   READY   UP-TO-DATE   AVAILABLE   AGE
app    3/3     3            3           23m
```

</details>

<details>
<summary>
Worker Node Failure
</summary>

```shell kubelet: Not Started
$ kubectl get node
NAME           STATUS     ROLES           AGE   VERSION
controlplane   Ready      control-plane   12m   v1.29.0
node01         NotReady   <none>          11m   v1.29.0
$ kubectl describe node node01 | grep -i condition -A10
Conditions:
  Type                 Status    LastHeartbeatTime                 LastTransitionTime                Reason              Message
  ----                 ------    -----------------                 ------------------                ------              -------
  NetworkUnavailable   False     Thu, 18 Apr 2024 12:40:39 +0000   Thu, 18 Apr 2024 12:40:39 +0000   FlannelIsUp         Flannel is running on this node
  MemoryPressure       Unknown   Thu, 18 Apr 2024 12:46:09 +0000   Thu, 18 Apr 2024 12:51:26 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
  DiskPressure         Unknown   Thu, 18 Apr 2024 12:46:09 +0000   Thu, 18 Apr 2024 12:51:26 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
  PIDPressure          Unknown   Thu, 18 Apr 2024 12:46:09 +0000   Thu, 18 Apr 2024 12:51:26 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
  Ready                Unknown   Thu, 18 Apr 2024 12:46:09 +0000   Thu, 18 Apr 2024 12:51:26 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
Addresses:
  InternalIP:  192.26.155.12
  Hostname:    node01
$ ssh node01
$ systemctl status kubelet
○ kubelet.service - kubelet: The Kubernetes Node Agent
     Loaded: loaded (/lib/systemd/system/kubelet.service; enabled; vendor preset: enabled)
    Drop-In: /usr/lib/systemd/system/kubelet.service.d
             └─10-kubeadm.conf
     Active: inactive (dead) since Thu 2024-04-18 13:07:32 UTC; 15s ago
       Docs: https://kubernetes.io/docs/
    Process: 2568 ExecStart=/usr/bin/kubelet $KUBELET_KUBECONFIG_ARGS $KUBELET_CONFIG_ARGS $KUBELET_KUBEADM_ARGS $KUBELET_EXTRA_>
   Main PID: 2568 (code=exited, status=0/SUCCESS)
...
$ systemctl restart kubelet
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   13m   v1.29.0
node01         Ready    <none>          12m   v1.29.0
```

```shell kubelet: CA File
$ kubectl get node
NAME           STATUS     ROLES           AGE   VERSION
controlplane   Ready      control-plane   15m   v1.29.0
node01         NotReady   <none>          14m   v1.29.0
$ kubectl describe node node01 | grep -i condition -A10
Conditions:
  Type                 Status    LastHeartbeatTime                 LastTransitionTime                Reason              Message
  ----                 ------    -----------------                 ------------------                ------              -------
  NetworkUnavailable   False     Thu, 18 Apr 2024 12:40:39 +0000   Thu, 18 Apr 2024 12:40:39 +0000   FlannelIsUp         Flannel is running on this node
  MemoryPressure       Unknown   Thu, 18 Apr 2024 12:52:47 +0000   Thu, 18 Apr 2024 12:54:31 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
  DiskPressure         Unknown   Thu, 18 Apr 2024 12:52:47 +0000   Thu, 18 Apr 2024 12:54:31 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
  PIDPressure          Unknown   Thu, 18 Apr 2024 12:52:47 +0000   Thu, 18 Apr 2024 12:54:31 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
  Ready                Unknown   Thu, 18 Apr 2024 12:52:47 +0000   Thu, 18 Apr 2024 12:54:31 +0000   NodeStatusUnknown   Kubelet stopped posting node status.
Addresses:
  InternalIP:  192.26.155.12
  Hostname:    node01
$ ssh node01
$ journalctl -u kubelet -f
...
Apr 18 12:57:39 node01 kubelet[9974]: E0418 12:57:39.113893    9974 run.go:74] "command failed" err="failed to construct kubelet dependencies: unable to load client CA file /etc/kubernetes/pki/WRONG-CA-FILE.crt: open /etc/kubernetes/pki/WRONG-CA-FILE.crt: no such file or directory"
$ vi /var/lib/kubelet/config.yaml
apiVersion: kubelet.config.k8s.io/v1beta1
authentication:
  anonymous:
    enabled: false
  webhook:
    cacheTTL: 0s
    enabled: true
  x509:
    clientCAFile: /etc/kubernetes/pki/ca.crt
...
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   19m   v1.29.0
node01         Ready    <none>          18m   v1.29.0
```

```shell kubelet: Port
$ kubectl get node
NAME           STATUS     ROLES           AGE   VERSION
controlplane   Ready      control-plane   20m   v1.29.0
node01         NotReady   <none>          20m   v1.29.0
$ ssh node01
$ journalctl -u kubelet | tail -n1
Apr 18 13:03:00 node01 kubelet[11012]: E0418 13:03:00.394377   11012 reflector.go:147] vendor/k8s.io/client-go/informers/factory.go:159: Failed to watch *v1.Node: failed to list *v1.Node: Get "https://controlplane:6553/api/v1/nodes?fieldSelector=metadata.name%3Dnode01&limit=500&resourceVersion=0": dial tcp 192.26.155.8:6553: connect: connection refused
$ vi /etc/kubernetes/kubelet.conf
apiVersion: v1
clusters:
- cluster:
...
    server: https://controlplane:6443
...
$ systemctl restart kubelet
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   26m   v1.29.0
node01         Ready    <none>          25m   v1.29.0
```

</details>

<details>
<summary>
Troubleshoot Network
</summary>

```shell Weave Net
$ kubectl get -n triton all
NAME                                READY   STATUS              RESTARTS   AGE
pod/mysql                           0/1     ContainerCreating   0          45s
pod/webapp-mysql-689d7dc744-8dbgn   0/1     ContainerCreating   0          45s

NAME                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql         ClusterIP   10.111.212.31   <none>        3306/TCP         45s
service/web-service   NodePort    10.96.217.141   <none>        8080:30081/TCP   44s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   0/1     1            0           45s

NAME                                      DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-689d7dc744   1         1         0       45s
$ kubectl describe -n triton po | grep -i event -A10
Events:
  Type     Reason                  Age                  From               Message
  ----     ------                  ----                 ----               -------
  Normal   Scheduled               2m31s                default-scheduler  Successfully assigned triton/mysql to controlplane
  Warning  FailedCreatePodSandBox  2m31s                kubelet            Failed to create pod sandbox: rpc error: code = Unknown desc = failed to setup network for sandbox "7b69000c1f2e1514604f21d29b0110b476df4f51860a401c140c70a0e0c027b7": plugin type="weave-net" name="weave" failed (add): unable to allocate IP address: Post "http://127.0.0.1:6784/ip/7b69000c1f2e1514604f21d29b0110b476df4f51860a401c140c70a0e0c027b7": dial tcp 127.0.0.1:6784: connect: connection refused
  Normal   SandboxChanged          4s (x12 over 2m30s)  kubelet            Pod sandbox changed, it will be killed and re-created.


Name:             webapp-mysql-689d7dc744-8dbgn
Namespace:        triton
Priority:         0
--
Events:
  Type     Reason                  Age                  From               Message
  ----     ------                  ----                 ----               -------
  Normal   Scheduled               2m30s                default-scheduler  Successfully assigned triton/webapp-mysql-689d7dc744-8dbgn to controlplane
  Warning  FailedCreatePodSandBox  2m30s                kubelet            Failed to create pod sandbox: rpc error: code = Unknown desc = failed to setup network for sandbox "3fca38780216071a794a0e4b4a12ac8e855a8ba238802a19983f0d2e86128d83": plugin type="weave-net" name="weave" failed (add): unable to allocate IP address: Post "http://127.0.0.1:6784/ip/3fca38780216071a794a0e4b4a12ac8e855a8ba238802a19983f0d2e86128d83": dial tcp 127.0.0.1:6784: connect: connection refused
  Normal   SandboxChanged          6s (x12 over 2m29s)  kubelet            Pod sandbox changed, it will be killed and re-created.
$ curl -L https://github.com/weaveworks/weave/releases/download/latest_release/weave-daemonset-k8s-1.11.yaml | kubectl apply -f -
$ kubectl get -n triton all
NAME                                READY   STATUS    RESTARTS   AGE
pod/mysql                           1/1     Running   0          6m59s
pod/webapp-mysql-689d7dc744-8dbgn   1/1     Running   0          6m59s

NAME                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql         ClusterIP   10.111.212.31   <none>        3306/TCP         6m59s
service/web-service   NodePort    10.96.217.141   <none>        8080:30081/TCP   6m58s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           6m59s

NAME                                      DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-689d7dc744   1         1         1       6m59s
```

```shell kube-proxy
$ kubectl get -n triton all
NAME                                READY   STATUS    RESTARTS   AGE
pod/mysql                           1/1     Running   0          56s
pod/webapp-mysql-689d7dc744-c82rm   1/1     Running   0          56s

NAME                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/mysql         ClusterIP   10.111.212.31   <none>        3306/TCP         8m49s
service/web-service   NodePort    10.96.217.141   <none>        8080:30081/TCP   8m48s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/webapp-mysql   1/1     1            1           56s

NAME                                      DESIRED   CURRENT   READY   AGE
replicaset.apps/webapp-mysql-689d7dc744   1         1         1       56s
$ kubectl get -n kube-system po
NAME                                   READY   STATUS             RESTARTS      AGE
coredns-76f75df574-d4gfv               1/1     Running            0             20m
coredns-76f75df574-fzbn7               1/1     Running            0             20m
etcd-controlplane                      1/1     Running            0             20m
kube-apiserver-controlplane            1/1     Running            0             20m
kube-controller-manager-controlplane   1/1     Running            0             20m
kube-proxy-62rk8                       0/1     CrashLoopBackOff   5 (97s ago)   4m29s
kube-scheduler-controlplane            1/1     Running            0             20m
weave-net-qmddf                        2/2     Running            0             5m50s
$ kubectl edit -n kube-system ds kube-proxy
...
    spec:
      containers:
      - command:
        - /usr/local/bin/kube-proxy
        - --config=/var/lib/kube-proxy/config.conf
        - --hostname-override=$(NODE_NAME)
...
```

</details>

## Mock Exam

<details>
<summary>
Warming Up
</summary>

```shell
$ kubectl run nginx-pod --image nginx:alpine
pod/nginx-pod created
$ kubectl run messaging --image redis:alpine -l tier=msg
pod/messaging created
$ kubectl create ns apx-x9984574
namespace/apx-x9984574 created
$ kubectl get node -ojson > /opt/outputs/nodes-z3444kd9.json
$ kubectl expose po messaging --name=messaging-service --port 6379
service/messaging-service exposed
$ kubectl create deploy hr-web-app --image kodekloud/webapp-color --replicas 2
deployment.apps/hr-web-app created
$ kubectl run static-busybox --image busybox --dry-run=client -oyaml > /etc/kubernetes/manifests/tmp.yaml
$ kubectl run -n finance temp-bus --image redis:alpine
pod/temp-bus created
$ kubectl get po orange -oyaml > tmp.yaml
$ vi tmp.yaml
$ kubectl replace -f tmp.yaml --force
pod "orange" deleted
pod/orange replaced
$ kubectl expose deploy hr-web-app --type NodePort --name hr-web-app-service --port 8080 --dry-run=client -oyaml > tmp.yaml
$ kubectl apply -f tmp.yaml
service/hr-web-app-service created
```

</details>

<details>
<summary>
<code>kubectl get -o</code>
</summary>

```shell Custom Columns
$ kubectl get -n admin2406 deploy -ojson | jq -c paths | grep name
["items",0,"metadata","name"]
...
$ kubectl get -n admin2406 deploy -ojson | jq -c paths | grep image
["items",0,"spec","template","spec","containers",0,"image"]
...
$ kubectl get -n admin2406 deploy -ojson | jq -c paths | grep replica
["items",0,"spec","replicas"]
...
$ kubectl get -n admin2406 deploy -ojson | jq -c paths | grep namespace
["items",0,"metadata","namespace"]
...
$ kubectl get -n admin2406 deploy -ocustom-columns=DEPLOYMENT:.metadata.name,CONTAINER_IMAGE:.spec.template.spec.containers[].image,READY_REPLICAS:.spec.replicas,NAMESPACE:.metadata.namespace --sort-by .metadata.name
DEPLOYMENT   CONTAINER_IMAGE   READY_REPLICAS   NAMESPACE
deploy1      nginx             1                admin2406
deploy2      nginx:alpine      1                admin2406
deploy3      nginx:1.16        1                admin2406
deploy4      nginx:1.17        1                admin2406
deploy5      nginx:latest      1                admin2406
$ kubectl get -n admin2406 deploy -ocustom-columns=DEPLOYMENT:.metadata.name,CONTAINER_IMAGE:.spec.template.spec.containers[].image,READY_REPLICAS:.spec.replicas,NAMESPACE:.metadata.namespace --sort-by .metadata.name > /opt/admin2406_data
```

```shell Json Path
$ kubectl get node -ojson | jq -c paths | grep osImage
["items",0,"status","nodeInfo","osImage"]
$ kubectl get node -ojsonpath={.items[*].status.nodeInfo.osImage}
Ubuntu 22.04.4 LTS
$ kubectl get node -ojsonpath={.items[*].status.nodeInfo.osImage} > /opt/outputs/nodes_os_x43kj56.txt
```

```shell Json Path
$ kubectl get node -ojson | jq -c paths | grep type | grep -v conditions
["items",0,"status","addresses",0,"type"]
["items",0,"status","addresses",1,"type"]
["items",1,"status","addresses",0,"type"]
["items",1,"status","addresses",1,"type"]
$ kubectl get node -ojsonpath='{.items[*].status.addresses[?(@.type=="InternalIP")].address}'
192.23.73.6 192.23.73.9
$ kubectl get node -ojsonpath='{.items[*].status.addresses[?(@.type=="InternalIP")].address}' > /root/CKA/node_ips
```

</details>

<details>
<summary>
Cluster Upgrade
</summary>

```shell
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   97m   v1.28.0
node01         Ready    <none>          97m   v1.28.0
```

```shell Upgrade (Control Plane)
$ echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /
$ apt update
$ apt-cache madison kubeadm
   kubeadm | 1.29.4-2.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt-mark unhold kubeadm
$ apt-get update
$ apt-get install -y kubeadm=1.29.0-1.1
$ apt-mark hold kubeadm
$ kubeadm upgrade apply v1.29.0
...
[upgrade/successful] SUCCESS! Your cluster was upgraded to "v1.29.0". Enjoy!
...
$ kubectl drain controlplane --ignore-daemonsets
node/controlplane cordoned
Warning: ignoring DaemonSet-managed Pods: kube-flannel/kube-flannel-ds-fwjlg, kube-system/kube-proxy-g6hct
evicting pod kube-system/coredns-5dd5756b68-hv2qj
evicting pod default/blue-667bf6b9f9-d87ww
evicting pod default/blue-667bf6b9f9-qxslb
pod/blue-667bf6b9f9-d87ww evicted
pod/blue-667bf6b9f9-qxslb evicted
pod/coredns-5dd5756b68-hv2qj evicted
node/controlplane drained
$ apt-mark unhold kubelet kubectl
$ apt-get update
$ apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
$ apt-mark hold kubelet kubectl
$ systemctl daemon-reload
$ systemctl restart kubelet
$ kubectl uncordon controlplane
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   56m   v1.29.0
node01         Ready    <none>          56m   v1.28.0
```

```shell Upgrade (Worker Node)
$ ssh node01
$ echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /
$ apt update
$ apt-cache madison kubeadm
   kubeadm | 1.29.4-2.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
   kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt-mark unhold kubeadm
$ apt-get update
$ apt-get install -y kubeadm=1.29.0-1.1
$ apt-mark hold kubeadm
$ kubeadm upgrade node
[upgrade] Reading configuration from the cluster...
[upgrade] FYI: You can look at this config file with 'kubectl -n kube-system get cm kubeadm-config -o yaml'
[preflight] Running pre-flight checks
[preflight] Skipping prepull. Not a control plane node.
[upgrade] Skipping phase. Not a control plane node.
[upgrade] Backing up kubelet config file to /etc/kubernetes/tmp/kubeadm-kubelet-config3882026834/config.yaml
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[upgrade] The configuration for this node was successfully updated!
[upgrade] Now you should go ahead and upgrade the kubelet package using your package manager.
$ kubectl drain node01 --ignore-daemonsets
node/node01 cordoned
Warning: ignoring DaemonSet-managed Pods: kube-flannel/kube-flannel-ds-ckbls, kube-system/kube-proxy-xjk9s
evicting pod kube-system/coredns-76f75df574-t5tzt
evicting pod default/blue-667bf6b9f9-s4784
evicting pod default/blue-667bf6b9f9-fwjvr
evicting pod default/blue-667bf6b9f9-z9wdm
evicting pod kube-system/coredns-76f75df574-8j4bn
evicting pod default/blue-667bf6b9f9-gcfbj
evicting pod default/blue-667bf6b9f9-z5d9w
pod/blue-667bf6b9f9-fwjvr evicted
pod/blue-667bf6b9f9-z9wdm evicted
pod/blue-667bf6b9f9-gcfbj evicted
pod/blue-667bf6b9f9-s4784 evicted
pod/blue-667bf6b9f9-z5d9w evicted
pod/coredns-76f75df574-t5tzt evicted
pod/coredns-76f75df574-8j4bn evicted
node/node01 drained
$ apt-mark unhold kubelet kubectl
$ apt-get update
$ apt-get install -y kubelet=1.29.0-1.1 kubectl=1.29.0-1.1
$ apt-mark hold kubelet kubectl
$ systemctl daemon-reload
$ systemctl restart kubelet
$ kubectl uncordon node01
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   63m   v1.29.0
node01         Ready    <none>          62m   v1.29.0
```

</details>

<details>
<summary>
etcd Backup
</summary>

```shell
$ kubectl get -n kube-system po etcd-controlplane -oyaml | grep ca
    - --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
    - --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
  priorityClassName: system-node-critical
  - containerID: containerd://47ff29cb7e3294dd7730bdce26ebff3272c6763ca18489d4fd5ec8982bab4d7d
$ ca=/etc/kubernetes/pki/etcd/ca.crt
$ kubectl get -n kube-system po etcd-controlplane -oyaml | grep cert
    - --cert-file=/etc/kubernetes/pki/etcd/server.crt
    - --client-cert-auth=true
    - --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
    - --peer-client-cert-auth=true
      name: etcd-certs
    name: etcd-certs
$ cert=/etc/kubernetes/pki/etcd/server.crt
$ kubectl get -n kube-system po etcd-controlplane -oyaml | grep key
    - --key-file=/etc/kubernetes/pki/etcd/server.key
    - --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
$ key=/etc/kubernetes/pki/etcd/server.key
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 --cacert=$ca --cert=$cert --key=$key snapshot save /opt/etcd-backup.db
```

</details>

<details>
<summary>
Pod: Mount Secret
</summary>

```shell
kubectl run -n admin1401 secret-1401 --image busybox --dry-run=client -oyaml > tmp.yaml
vi tmp.yaml
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: secret-1401
  name: secret-1401
  namespace: admin1401
spec:
  containers:
    - image: busybox
      name: secret-admin
      command:
        - sleep
        - "4800"
      volumeMounts:
        - mountPath: /etc/secret-volume
          name: secret-volume
          readOnly: true
  volumes:
    - name: secret-volume
      secret:
        secretName: dotfile-secret
```

</details>

<details>
<summary>
Pod: Mount Volume
</summary>

```shell
kubectl run redis-storage --image redis:alpine --dry-run=client -oyaml > tmp.yaml
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: redis-storage
  name: redis-storage
spec:
  containers:
    - image: redis:alpine
      name: redis-storage
      volumeMounts:
        - mountPath: /data/redis
          name: cache-volume
  volumes:
    - name: cache-volume
      emptyDir:
        sizeLimit: 500Mi
```

```shell
$ kubectl apply -f tmp.yaml
pod/redis-storage created
```

</details>

<details>
<summary>
Pod: <code>securityContext</code>
</summary>

```shell
kubectl run super-user-pod --image busybox:1.28 --dry-run=client -oyaml > tmp.yaml
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: super-user-pod
  name: super-user-pod
spec:
  containers:
    - image: busybox:1.28
      name: super-user-pod
      securityContext:
        capabilities:
          add: ["SYS_TIME"]
```

```shell
$ kubectl apply -f tmp.yaml
pod/super-user-pod created
```

</details>

<details>
<summary>
PV & PVC
</summary>

```shell
$ kubectl get pv
NAME   CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      CLAIM   STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-1   10Mi       RWO            Retain           Available                          <unset>                          6m36s
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
persistentvolumeclaim/my-pvc created
$ vi CKA/use-pv.yaml
$ kubectl apply -f CKA/use-pv.yaml
pod/use-pv created
```

```yaml tmp.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Mi
```

```yaml /root/CKA/use-pv.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: use-pv
  name: use-pv
spec:
  containers:
    - image: nginx
      name: use-pv
      volumeMounts:
        - mountPath: "/data"
          name: mypv
  volumes:
    - name: mypv
      persistentVolumeClaim:
        claimName: my-pvc
```

</details>

<details>
<summary>
Deployment: Create & Upgrade
</summary>

```shell
$ kubectl create deploy nginx-deploy --image nginx:1.16
deployment.apps/nginx-deploy created
$ kubectl edit deploy nginx-deploy
...
    spec:
      containers:
        - image: nginx:1.17
...
deployment.apps/nginx-deploy edited
```

</details>

<details>
<summary>
CertificateSigningRequest
</summary>

```shell
$ cat john.csr | base64 -w0
LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQ1ZEQ0NBVHdDQVFBd0R6RU5NQXNHQTFVRUF3d0VhbTlvYmpDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRApnZ0VQQURDQ0FRb0NnZ0VCQUpWSzFITXo4R2hUVk42RnN3MHdMM2VQUE5EQjRObWRuSHNPMkVFTVVsZ0xOTCtpClJieHVJMUkvMFhoOTZ4eVl6Z2FEblpoZk9CMWsraEtDYnFmSGMzTmNrT0NqdFBaUWhsSzdJN0ZiQ08wRGhYa0oKTmVPMmJEY1N6Z1ZxbHFZM1NNV0JYWHlxSk5DMjBpZ1FGWHZmeFdCRVZQS1lEb0tycHpNNmRVZUdyMTNsbkhIZAord2VHSUkwbGVLYXhBNFh5ZXA2Sk40elZqdndHaFZPSjVRSXUrNUhUYUg1Sm1NeVBlY1RiN1pRVzNIbzIzNHpYClpsdkQxam1rS2t4R2ZuV0pJOFFYemk5UjAwekcrdnVKcXlPNTVZRUlFMUxzbkN2UGFCMWovbmR0OEtJUjVOZEEKSEJieENQYUpZNEFBV3JaUG9RMjdWekpuZFUzT1RWSllNektuaStrQ0F3RUFBYUFBTUEwR0NTcUdTSWIzRFFFQgpDd1VBQTRJQkFRQkRodXdXS3c1RVpxaG5PSDZYbmxDeVRzZUptcUY5OGorUXBjMmVZWWtQSUFMeGNQV0sySGNSCjgzNlN1aTVOMDB6NlJNT0RwLzVpNU0yaUU1NFd2VzU2ZmpVL1Y1eTRJRVR1eC9hSk12ZmpETkxVZDN5SkVoaS8KdC81UkhZNG5Gbk8xMjhuTVhaVWVQS0lPS085MTVmWTgvUmVmV04vVWNoajhIL243aXZRZVlmdVp2RlNHU2RxTwpJY3FCRkZ0WWxzZVZSTDZLeHlpUldPRlBwNUpZWFdUZ3FBVzgwNGpNS2luSldnRk10WXVtaFpFZmgwTUlXSDYvCm8zdk1qTktWUXZ3NlE0RG5SMzBFZE02bU1JRHdLbWJjb053L2lmMUVJWXJGMHAzUGg2UG50RkdTVGZOQ2EyS1IKN2JOaThGcW9mZVVNUCtNZmpkSENZeGdCQ1psT0FRc0cKLS0tLS1FTkQgQ0VSVElGSUNBVEUgUkVRVUVTVC0tLS0tCg==
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
certificatesigningrequest.certificates.k8s.io/john-developer created
$ kubectl certificate approve john-developer
$ kubectl create -n development role developer --resource po --verb create,list,get,update,delete
role.rbac.authorization.k8s.io/developer created
$ kubectl create -n development rolebinding developer-role-binding --role developer --user john
rolebinding.rbac.authorization.k8s.io/developer-role-binding created
$ kubectl auth -n development can-i update po --as john
yes
```

```yaml tmp.yaml
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: john-developer
spec:
  request: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQ1ZEQ0NBVHdDQVFBd0R6RU5NQXNHQTFVRUF3d0VhbTlvYmpDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRApnZ0VQQURDQ0FRb0NnZ0VCQUpWSzFITXo4R2hUVk42RnN3MHdMM2VQUE5EQjRObWRuSHNPMkVFTVVsZ0xOTCtpClJieHVJMUkvMFhoOTZ4eVl6Z2FEblpoZk9CMWsraEtDYnFmSGMzTmNrT0NqdFBaUWhsSzdJN0ZiQ08wRGhYa0oKTmVPMmJEY1N6Z1ZxbHFZM1NNV0JYWHlxSk5DMjBpZ1FGWHZmeFdCRVZQS1lEb0tycHpNNmRVZUdyMTNsbkhIZAord2VHSUkwbGVLYXhBNFh5ZXA2Sk40elZqdndHaFZPSjVRSXUrNUhUYUg1Sm1NeVBlY1RiN1pRVzNIbzIzNHpYClpsdkQxam1rS2t4R2ZuV0pJOFFYemk5UjAwekcrdnVKcXlPNTVZRUlFMUxzbkN2UGFCMWovbmR0OEtJUjVOZEEKSEJieENQYUpZNEFBV3JaUG9RMjdWekpuZFUzT1RWSllNektuaStrQ0F3RUFBYUFBTUEwR0NTcUdTSWIzRFFFQgpDd1VBQTRJQkFRQkRodXdXS3c1RVpxaG5PSDZYbmxDeVRzZUptcUY5OGorUXBjMmVZWWtQSUFMeGNQV0sySGNSCjgzNlN1aTVOMDB6NlJNT0RwLzVpNU0yaUU1NFd2VzU2ZmpVL1Y1eTRJRVR1eC9hSk12ZmpETkxVZDN5SkVoaS8KdC81UkhZNG5Gbk8xMjhuTVhaVWVQS0lPS085MTVmWTgvUmVmV04vVWNoajhIL243aXZRZVlmdVp2RlNHU2RxTwpJY3FCRkZ0WWxzZVZSTDZLeHlpUldPRlBwNUpZWFdUZ3FBVzgwNGpNS2luSldnRk10WXVtaFpFZmgwTUlXSDYvCm8zdk1qTktWUXZ3NlE0RG5SMzBFZE02bU1JRHdLbWJjb053L2lmMUVJWXJGMHAzUGg2UG50RkdTVGZOQ2EyS1IKN2JOaThGcW9mZVVNUCtNZmpkSENZeGdCQ1psT0FRc0cKLS0tLS1FTkQgQ0VSVElGSUNBVEUgUkVRVUVTVC0tLS0tCg==
  signerName: kubernetes.io/kube-apiserver-client
  usages:
    - client auth
```

</details>

<details>
<summary>
Pod: Service
</summary>

```shell
$ kubectl run nginx-resolver --image nginx
pod/nginx-resolver created
$ kubectl expose po nginx-resolver --name nginx-resolver-service --port 80
service/nginx-resolver-service exposed
$ kubectl run test-nslookup --image busybox:1.28 --rm -it --restart=Never -- nslookup nginx-resolver-service
Server:    10.96.0.10
Address 1: 10.96.0.10 kube-dns.kube-system.svc.cluster.local

Name:      nginx-resolver-service
Address 1: 10.111.187.86 nginx-resolver-service.default.svc.cluster.local
pod "test-nslookup" deleted
$ kubectl run test-nslookup --image busybox:1.28 --rm -it --restart=Never -- nslookup nginx-resolver-service > /root/CKA/nginx.svc
$ kubectl get pod nginx-resolver -o wide
NAME             READY   STATUS    RESTARTS   AGE   IP             NODE     NOMINATED NODE   READINESS GATES
nginx-resolver   1/1     Running   0          97s   10.244.192.4   node01   <none>           <none>
$ kubectl run test-nslookup --image busybox:1.28 --rm -it --restart Never -- nslookup 10.244.192.4
Server:    10.96.0.10
Address 1: 10.96.0.10 kube-dns.kube-system.svc.cluster.local

Name:      10.244.192.4
Address 1: 10.244.192.4 10-244-192-4.nginx-resolver-service.default.svc.cluster.local
pod "test-nslookup" deleted
$ kubectl run test-nslookup --image busybox:1.28 --rm -it --restart Never -- nslookup 10.244.192.4 > /root/CKA/nginx.pod
```

</details>

<details>
<summary>
Static Pod
</summary>

```shell
kubectl run nginx-critical --image nginx --dry-run=client -oyaml > tmp.yaml
ssh node01
vi /etc/kubernetes/manifests/tmp.yaml
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: nginx-critical
  name: nginx-critical
spec:
  containers:
    - image: nginx
      name: nginx-critical
```

</details>

<details>
<summary>
ServiceAccount
</summary>

```shell
$ kubectl create sa pvviewer
serviceaccount/pvviewer created
$ kubectl create clusterrole pvviewer-role --resource pv --verb list
clusterrole.rbac.authorization.k8s.io/pvviewer-role created
$ kubectl create clusterrolebinding pvviewer-role-binding --clusterrole pvviewer-role --serviceaccount default:pvviewer
clusterrolebinding.rbac.authorization.k8s.io/pvviewer-role-binding created
$ kubectl run pvviewer --image redis --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/pvviewer created
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: pvviewer
  name: pvviewer
spec:
  serviceAccountName: pvviewer
  containers:
    - image: redis
      name: pvviewer
```

</details>

<details>
<summary>
Pod: Multi-container
</summary>

```shell
$ kubectl run multi-pod --image nginx --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/multi-pod created
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: multi-pod
  name: multi-pod
spec:
  containers:
  - image: nginx
    name: alpha
    env:
      - name: name
        value: alpha
  - image: busybox
    name: beta
    command:
      - sleep
      - "4800"
    env:
      - name: name
        value: beta
```

</details>

<details>
<summary>
NetworkPolicy
</summary>

```shell
$ kubectl run test --image alpine/curl --rm -it --restart Never -- sh
If you don't see a command prompt, try pressing enter.
/ # curl np-test-service
curl: (28) Failed to connect to np-test-service port 80 after 131147 ms: Couldn't connect to server
pod "test" deleted
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
networkpolicy.networking.k8s.io/ingress-to-nptest created
$ kubectl run test --image alpine/curl --rm -it --restart Never -- sh
If you don't see a command prompt, try pressing enter.
/ # curl np-test-service
<!DOCTYPE html>
...
pod "test" deleted
```

```yaml tmp.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ingress-to-nptest
spec:
  podSelector:
    matchLabels:
      run: np-test-1
  policyTypes:
    - Ingress
  ingress:
    - ports:
        - protocol: TCP
          port: 80
```

</details>

<details>
<summary>
Node: Taint
</summary>

```shell
$ kubectl taint node node01 env_type=production:NoSchedule
node/node01 tainted
$ kubectl run dev-redis --image redis:alpine
pod/dev-redis created
$ kubectl get po dev-redis -owide
NAME        READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
dev-redis   1/1     Running   0          13s   10.244.0.4   controlplane   <none>           <none>
$ kubectl run prod-redis --image redis:alpine --dry-run=client -oyaml > tmp.yaml
$ kubectl get po prod-redis -owide
NAME         READY   STATUS    RESTARTS   AGE   IP             NODE     NOMINATED NODE   READINESS GATES
prod-redis   1/1     Running   0          57s   10.244.192.5   node01   <none>           <none>
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  name: prod-redis
spec:
  containers:
    - name: prod-redis
      image: redis:alpine
  tolerations:
    - key: env_type
      operator: Equal
      value: production
      effect: NoSchedule
```

</details>

## Killer Shell

<details>
<summary>
Contexts
</summary>

```shell
$ kubectl config get-contexts -oname
k8s-c1-H
k8s-c2-AC
k8s-c3-CCC
$ kubectl config get-contexts -oname > /opt/course/1/contexts
$ kubectl config current-context
k8s-c1-H
$ echo 'kubectl config current-context' > /opt/course/1/context_default_kubectl.sh
$ cat ~/.kube/config | grep current | sed 's/current-context: //'
k8s-c1-H
$ echo "cat ~/.kube/config | grep current | sed 's/current-context: //'" > /opt/course/1/context_default_no_kubectl.sh
```

</details>

<details>
<summary>
Schedule Pod on Control Plane
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl run pod1 --image httpd:2.4.41-alpine --dry-run=client -oyaml > tmp.yaml
$ kubectl describe node cluster1-controlplane1 | grep Taint
Taints:             node-role.kubernetes.io/control-plane:NoSchedule
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/pod1 created
$ kubectl get po pod1 -owide
NAME   READY   STATUS    RESTARTS   AGE   IP          NODE                     NOMINATED NODE   READINESS GATES
pod1   1/1     Running   0          18s   10.32.0.2   cluster1-controlplane1   <none>           <none>
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: pod1
  name: pod1
spec:
  containers:
    - image: httpd:2.4.41-alpine
      name: pod1-container
  tolerations:
    - key: node-role.kubernetes.io/control-plane
      operator: Exists
      effect: NoSchedule
```

</details>

<details>
<summary>
Scale down StatefulSet
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl get -n project-c13 po | grep o3db
o3db-0                                 1/1     Running   0          119d
o3db-1                                 1/1     Running   0          119d
$ kubectl get -n project-c13 sts
NAME   READY   AGE
o3db   2/2     119d
$ kubectl scale -n project-c13 sts o3db --replicas 1
statefulset.apps/o3db scaled
$ kubectl get -n project-c13 po | grep o3db
o3db-0                                 1/1     Running   0          119d
```

</details>

<details>
<summary>
Pod Ready if Service is Reachable
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl run ready-if-service-ready --image nginx:1.16.1-alpine --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/ready-if-service-ready created
$ kubectl run am-i-ready --image nginx:1.16.1-alpine -l id=cross-server-ready
pod/am-i-ready created
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: ready-if-service-ready
  name: ready-if-service-ready
spec:
  containers:
    - image: nginx:1.16.1-alpine
      name: ready-if-service-ready
      livenessProbe:
        exec:
          command:
            - "true"
      readinessProbe:
        exec:
          command:
            - sh
            - -c
            - wget -T2 -O- http://service-am-i-ready:80
```

</details>

<details>
<summary>
kubectl sorting
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl get -A po --sort-by metadata.creationTimestamp
NAMESPACE         NAME                                             READY   STATUS      RESTARTS       AGE
kube-system       kube-controller-manager-cluster1-controlplane1   1/1     Running     0              119d
kube-system       kube-scheduler-cluster1-controlplane1            1/1     Running     0              119d
kube-system       etcd-cluster1-controlplane1                      1/1     Running     0              119d
...
$ echo 'kubectl get -A po --sort-by metadata.creationTimestamp' > /opt/course/5/find_pods.sh
$ kubectl get -A po --sort-by metadata.uid
NAMESPACE         NAME                                             READY   STATUS      RESTARTS       AGE
project-c14       c14-3cc-runner-d4478577-6984v                    1/1     Running     0              119d
project-c14       c14-3cc-runner-heavy-7f89fb68f5-gtwm2            1/1     Running     0              119d
project-c14       c14-3cc-runner-d4478577-xdx87                    1/1     Running     0              119d
...
$ echo 'kubectl get -A po --sort-by metadata.uid' > /opt/course/5/find_pods_uid.sh
```

</details>

<details>
<summary>
Storage, PV, PVC, Pod volume
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
persistentvolume/safari-pv created
persistentvolumeclaim/safari-pvc created
deployment.apps/safari created
$ kubectl get -n project-tiger deploy safari
NAME     READY   UP-TO-DATE   AVAILABLE   AGE
safari   1/1     1            1           40s
```

```yaml tmp.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: safari-pv
spec:
  capacity:
    storage: 2Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/Volumes/Data"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: safari-pvc
  namespace: project-tiger
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safari
  namespace: project-tiger
spec:
  replicas: 1
  selector:
    matchLabels:
      app: safari
  template:
    metadata:
      labels:
        app: safari
    spec:
      containers:
        - name: safari
          image: httpd:2.4.41-alpine
          volumeMounts:
            - mountPath: "/tmp/safari-data"
              name: safari-pvc
      volumes:
        - name: safari-pvc
          persistentVolumeClaim:
            claimName: safari-pvc
```

</details>

<details>
<summary>
Node and Pod Resource Usage
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl top node
NAME                     CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
cluster1-controlplane1   187m         9%     1121Mi          59%
cluster1-node1           84m          8%     797Mi           42%
cluster1-node2           66m          6%     790Mi           41%
$ echo 'kubectl top node' > /opt/course/7/node.sh
$ kubectl top po --containers=true
POD                          NAME                     CPU(cores)   MEMORY(bytes)
am-i-ready                   am-i-ready               1m           2Mi
multi-container-playground   c1                       0m           2Mi
multi-container-playground   c2                       2m           0Mi
multi-container-playground   c3                       1m           0Mi
pod1                         pod1-container           1m           5Mi
ready-if-service-ready       ready-if-service-ready   3m           2Mi
web-test-677cc57468-gj99x    httpd                    1m           5Mi
web-test-677cc57468-kqn42    httpd                    1m           5Mi
web-test-677cc57468-lzshp    httpd                    1m           4Mi
$ echo 'kubectl top po --containers=true' > /opt/course/7/pod.sh
```

</details>

<details>
<summary>
Get Control Plane Information
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ ssh cluster1-controlplane1
$ systemctl status kubelet
● kubelet.service - kubelet: The Kubernetes Node Agent
     Loaded: loaded (/lib/systemd/system/kubelet.service; enabled; vendor preset: enabled)
    Drop-In: /usr/lib/systemd/system/kubelet.service.d
             └─10-kubeadm.conf
...
$ kubectl get -n kube-system po
NAME                                             READY   STATUS    RESTARTS       AGE
coredns-5c645bd457-jd2tg                         1/1     Running   0              119d
coredns-5c645bd457-srjnd                         1/1     Running   0              119d
etcd-cluster1-controlplane1                      1/1     Running   0              119d
kube-apiserver-cluster1-controlplane1            1/1     Running   0              119d
kube-controller-manager-cluster1-controlplane1   1/1     Running   0              119d
kube-proxy-2f94g                                 1/1     Running   0              119d
kube-proxy-tdlbt                                 1/1     Running   0              119d
kube-proxy-wgmlr                                 1/1     Running   0              119d
kube-scheduler-cluster1-controlplane1            1/1     Running   0              119d
metrics-server-7cdc4ddd88-6dd7g                  1/1     Running   0              119d
weave-net-t2kch                                  2/2     Running   1 (119d ago)   119d
weave-net-v2gvp                                  2/2     Running   1 (119d ago)   119d
weave-net-wbsb4                                  2/2     Running   0              119d
$ vi /opt/course/8/controlplane-components.txt
```

```yaml /opt/course/8/controlplane-components.txt
# /opt/course/8/controlplane-components.txt
kubelet: process
kube-apiserver: static-pod
kube-scheduler: static-pod
kube-controller-manager: static-pod
etcd: static-pod
dns: pod coredns
```

</details>

<details>
<summary>
Kill Scheduler, Manual Scheduling
</summary>

```shell
$ kubectl config use-context k8s-c2-AC
Switched to context "k8s-c2-AC".
$ ssh cluster2-controlplane1
$ mv /etc/kubernetes/manifests/kube-scheduler.yaml ./
$ kubectl get -n kube-system po kube-scheduler-cluster2-controlplane1
Error from server (NotFound): pods "kube-scheduler-cluster2-controlplane1" not found
$ kubectl run manual-schedule --image httpd:2.4-alpine --dry-run=client -oyaml > tmp.yaml
$ kubectl get node cluster2-controlplane1 --show-labels
NAME                     STATUS   ROLES           AGE    VERSION   LABELS
cluster2-controlplane1   Ready    control-plane   119d   v1.29.0   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/os=linux,kubernetes.io/arch=amd64,kubernetes.io/hostname=cluster2-controlplane1,kubernetes.io/os=linux,node-role.kubernetes.io/control-plane=,node.kubernetes.io/exclude-from-external-load-balancers=
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/manual-schedule created
$ kubectl get po manual-schedule -owide
NAME              READY   STATUS    RESTARTS   AGE   IP          NODE                     NOMINATED NODE   READINESS GATES
manual-schedule   1/1     Running   0          18s   10.32.0.2   cluster2-controlplane1   <none>           <none>
$ mv kube-scheduler.yaml /etc/kubernetes/manifests/
$ kubectl get -n kube-system po kube-scheduler-cluster2-controlplane1
NAME                                    READY   STATUS    RESTARTS   AGE
kube-scheduler-cluster2-controlplane1   1/1     Running   0          21s
$ kubectl run manual-schedule2 --image httpd:2.4-alpine
pod/manual-schedule2 created
$ kubectl get po manual-schedule2 -owide
NAME               READY   STATUS    RESTARTS   AGE   IP          NODE             NOMINATED NODE   READINESS GATES
manual-schedule2   1/1     Running   0          9s    10.44.0.3   cluster2-node1   <none>           <none>
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: manual-schedule
  name: manual-schedule
spec:
  containers:
    - image: httpd:2.4-alpine
      name: manual-schedule
  nodeName: cluster2-controlplane1
```

</details>

<details>
<summary>
RBAC ServiceAccount Role RoleBinding
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl create -n project-hamster sa processor
serviceaccount/processor created
$ kubectl create -n project-hamster role processor --resource secret,cm --verb create
role.rbac.authorization.k8s.io/processor created
$ kubectl create -n project-hamster rolebinding processor --role processor --serviceaccount project-hamster:processor
rolebinding.rbac.authorization.k8s.io/processor created
```

</details>

<details>
<summary>
DaemonSet on all Nodes
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl describe node  | grep Taint
Taints:             node-role.kubernetes.io/control-plane:NoSchedule
Taints:             <none>
Taints:             <none>
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
daemonset.apps/ds-important created
$ kubectl get -n project-tiger po -l id=ds-important -owide
$ kubectl get -n project-tiger po -l id=ds-important -owide
NAME                 READY   STATUS    RESTARTS   AGE   IP           NODE                     NOMINATED NODE   READINESS GATES
ds-important-2b7xt   1/1     Running   0          67s   10.36.0.23   cluster1-node2           <none>           <none>
ds-important-qkdrn   1/1     Running   0          67s   10.44.0.28   cluster1-node1           <none>           <none>
ds-important-z4ch9   1/1     Running   0          67s   10.32.0.3    cluster1-controlplane1   <none>           <none>
```

```yaml tmp.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ds-important
  namespace: project-tiger
  labels:
    id: ds-important
    uuid: 18426a0b-5f59-4e10-923f-c0e078e82462
spec:
  selector:
    matchLabels:
      id: ds-important
      uuid: 18426a0b-5f59-4e10-923f-c0e078e82462
  template:
    metadata:
      labels:
        id: ds-important
        uuid: 18426a0b-5f59-4e10-923f-c0e078e82462
    spec:
      tolerations:
        - effect: NoSchedule
          key: node-role.kubernetes.io/control-plane
      containers:
        - name: httpd
          image: httpd:2.4-alpine
          resources:
            limits:
              cpu: 10m
              memory: 10Mi
            requests:
              cpu: 10m
              memory: 10Mi
```

</details>

<details>
<summary>
Deployment on all Nodes
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl create -n project-tiger deploy deploy-important --image nginx:1.17.6-alpine --replicas 3 --dry-run=client -oyaml > tmp.yaml
$ kubectl apply -f tmp.yaml
deployment.apps/deploy-important created
$ kubectl get -n project-tiger po -l id=very-important -owide
NAME                                READY   STATUS    RESTARTS   AGE   IP           NODE             NOMINATED NODE   READINESS GATES
deploy-important-5759cd8b48-nnlxn   2/2     Running   0          15s   10.36.0.24   cluster1-node2   <none>           <none>
deploy-important-5759cd8b48-wrqr6   0/2     Pending   0          15s   <none>       <none>           <none>           <none>
deploy-important-5759cd8b48-zk8hk   2/2     Running   0          15s   10.44.0.29   cluster1-node1   <none>           <none>
```

```yaml tmp.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: deploy-important
    id: very-important
  name: deploy-important
  namespace: project-tiger
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deploy-important
      id: very-important
  template:
    metadata:
      labels:
        app: deploy-important
        id: very-important
    spec:
      containers:
        - image: nginx:1.17.6-alpine
          name: container1
        - image: google/pause
          name: container2
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: id
                    operator: In
                    values:
                      - very-important
              topologyKey: kubernetes.io/hostname
```

</details>

<details>
<summary>
Multi Containers and Pod shared Volume
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl run multi-container-playground --image nginx:1.17.6-alpine --dry-run=client -oyaml > tmp.yaml
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/multi-container-playground created
$ kubectl get po multi-container-playground
NAME                         READY   STATUS    RESTARTS   AGE
multi-container-playground   3/3     Running   0          17s
$ kubectl logs multi-container-playground -c c3
Thu May  2 14:00:06 UTC 2024
Thu May  2 14:00:07 UTC 2024
Thu May  2 14:00:08 UTC 2024
...
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: multi-container-playground
  name: multi-container-playground
spec:
  containers:
    - image: nginx:1.17.6-alpine
      name: c1
      env:
        - name: MY_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
      volumeMounts:
        - name: vol
          mountPath: /vol
    - image: busybox:1.31.1
      name: c2
      command:
        ["sh", "-c", "while true; do date >> /vol/date.log; sleep 1; done"]
      volumeMounts:
        - name: vol
          mountPath: /vol
    - image: busybox:1.31.1
      name: c3
      command: ["sh", "-c", "tail -f /vol/date.log"]
      volumeMounts:
        - name: vol
          mountPath: /vol
  volumes:
    - name: vol
      emptyDir: {}
```

</details>

<details>
<summary>
Find out Cluster Information
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl get node
NAME                     STATUS   ROLES           AGE    VERSION
cluster1-controlplane1   Ready    control-plane   119d   v1.29.0
cluster1-node1           Ready    <none>          119d   v1.29.0
cluster1-node2           Ready    <none>          119d   v1.29.0
$ kubectl describe -n kube-system po kube-apiserver-cluster1-controlplane1 | grep range
      --service-cluster-ip-range=10.96.0.0/12
$ ssh cluster1-controlplane1
$ ls /etc/cni/net.d/
10-weave.conflist
$ kubectl get -n kube-system po | grep controlplane
etcd-cluster1-controlplane1                      1/1     Running   0              119d
kube-apiserver-cluster1-controlplane1            1/1     Running   0              119d
kube-controller-manager-cluster1-controlplane1   1/1     Running   0              119d
kube-scheduler-cluster1-controlplane1            1/1     Running   0              119d
```

```yaml /opt/course/14/cluster-info
# /opt/course/14/cluster-info
1: 1
2: 2
3: 10.96.0.0/12
4: Weave, /etc/cni/net.d/10-weave.conflist
5: -cluster1-node1
```

</details>

<details>
<summary>
Cluster Event Logging
</summary>

```shell
$ kubectl config use-context k8s-c2-AC
Switched to context "k8s-c2-AC".
$ kubectl get -A events --sort-by metadata.creationTimestamp
NAMESPACE     LAST SEEN   TYPE      REASON           OBJECT                                      MESSAGE
kube-system   60m         Normal    Killing          pod/kube-scheduler-cluster2-controlplane1   Stopping container kube-scheduler
kube-system   58m         Warning   Failed           pod/kube-scheduler-cluster2-controlplane1   Error: ErrImagePull
kube-system   58m         Warning   Failed           pod/kube-scheduler-cluster2-controlplane1   Failed to pull image "registry.k8s.io/zerohertz": rpc error: code = NotFound desc = failed to pull and unpack image "registry.k8s.io/zerohertz:latest": failed to resolve reference "registry.k8s.io/zerohertz:latest": registry.k8s.io/zerohertz:latest: not found
...
$ echo 'kubectl get -A events --sort-by metadata.creationTimestamp' > /opt/course/15/cluster_events.sh
$ kubectl get -n kube-system po -owide | grep proxy
kube-proxy-5gwrf                                 1/1     Running   0              119d   192.168.100.21   cluster2-controlplane1   <none>           <none>
kube-proxy-f2l2m                                 1/1     Running   0              119d   192.168.100.22   cluster2-node1           <none>           <none>
$ kubectl delete -n kube-system po kube-proxy-f2l2m
pod "kube-proxy-f2l2m" deleted
$ kubectl get -A events --sort-by metadata.creationTimestamp | grep proxy
kube-system   23s         Normal   Killing            pod/kube-proxy-f2l2m                        Stopping container kube-proxy
kube-system   22s         Normal   Pulled             pod/kube-proxy-kbmhw                        Container image "registry.k8s.io/kube-proxy:v1.29.0" already present on machine
kube-system   22s         Normal   SuccessfulCreate   daemonset/kube-proxy                        Created pod: kube-proxy-kbmhw
kube-system   21s         Normal   Scheduled          pod/kube-proxy-kbmhw                        Successfully assigned kube-system/kube-proxy-kbmhw to cluster2-node1
kube-system   22s         Normal   Created            pod/kube-proxy-kbmhw                        Created container kube-proxy
kube-system   21s         Normal   Started            pod/kube-proxy-kbmhw                        Started container kube-proxy
$ kubectl get -A events --sort-by metadata.creationTimestamp | grep proxy > /opt/course/15/pod_kill.log
$ ssh cluster2-node1
$ crictl ps | grep proxy
edbeba078ec88       98262743b26f9       About a minute ago   Running             kube-proxy          0                   85920c4a3e434       kube-proxy-kbmhw
$ crictl stop edbeba078ec88
edbeba078ec88
$ crictl rm edbeba078ec88
edbeba078ec88
$ kubectl get -A events --sort-by metadata.creationTimestamp | grep proxy | tail -n3
kube-system   4m9s        Normal   Pulled             pod/kube-proxy-kbmhw                        Container image "registry.k8s.io/kube-proxy:v1.29.0" already present on machine
kube-system   4m8s        Normal   Started            pod/kube-proxy-kbmhw                        Started container kube-proxy
kube-system   4m9s        Normal   Created            pod/kube-proxy-kbmhw                        Created container kube-proxy
$ kubectl get -A events --sort-by metadata.creationTimestamp | grep proxy | tail -n3 > /opt/course/15/container_kill.log
```

</details>

<details>
<summary>
Namespace and API Resources
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl api-resources --namespaced -oname > /opt/course/16/resources.txt
$ kubectl get -n project-c14 role --no-headers | wc -l
300
$ kubectl get -n project-hamster role --no-headers | wc -l
1
$ echo 'project-c14 with 300 resources' > /opt/course/16/crowded-namespace.txt
```

</details>

<details>
<summary>
Find Container of Pod and check info
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl run -n project-tiger tigers-reunite --image httpd:2.4.41-alpine -l pod=container,container=pod
pod/tigers-reunite created
$ kubectl get -n project-tiger po tigers-reunite -owide
NAME             READY   STATUS    RESTARTS   AGE   IP           NODE             NOMINATED NODE   READINESS GATES
tigers-reunite   1/1     Running   0          28s   10.44.0.31   cluster1-node1   <none>           <none>
$ ssh cluster1-node1
$ crictl ps | grep tigers-reunite
e865f766c5384       54b0995a63052       About a minute ago   Running             tigers-reunite           0                   77f90134f631c       tigers-reunite
$ crictl inspect e865f766c5384 | grep runtimeType
    "runtimeType": "io.containerd.runc.v2",
$ echo 'e865f766c5384 io.containerd.runc.v2' > /opt/course/17/pod-container.txt
$ kubectl logs -n project-tiger tigers-reunite > /opt/course/17/pod-container.log
```

</details>

<details>
<summary>
Fix kubelet
</summary>

```shell
$ kubectl config use-context k8s-c3-CCC
Switched to context "k8s-c3-CCC".
$ ssh cluster3-node1
$ systemctl status kubelet
● kubelet.service - kubelet: The Kubernetes Node Agent
...
    Drop-In: /usr/lib/systemd/system/kubelet.service.d
             └─10-kubeadm.conf
...
    Process: 28725 ExecStart=/usr/local/bin/kubelet $KUBELET_KUBECONFIG_ARGS $K>
...
Jan 04 13:12:52 cluster3-node1 systemd[1]: kubelet.service: Main process exited>
Jan 04 13:12:52 cluster3-node1 systemd[1]: kubelet.service: Failed with result >
Jan 04 13:12:54 cluster3-node1 systemd[1]: Stopped kubelet: The Kubernetes Node>
$ ls /usr/local/bin/kubelet
ls: cannot access '/usr/local/bin/kubelet': No such file or directory
$ whereis kubelet
kubelet: /usr/bin/kubelet
$ sed -i "s|/usr/local/bin/kubelet|/usr/bin/kubelet|g" /usr/lib/systemd/system/kubelet.service.d/10-kubeadm.conf
$ systemctl daemon-reload
$ systemctl restart kubelet
$ systemctl status kubelet
● kubelet.service - kubelet: The Kubernetes Node Agent
     Loaded: loaded (/lib/systemd/system/kubelet.service; enabled; vendor preset: enabled)
    Drop-In: /usr/lib/systemd/system/kubelet.service.d
             └─10-kubeadm.conf
     Active: active (running) since Fri 2024-05-03 03:52:48 UTC; 4s ago
...
$ echo 'wrong path to kubelet binary specified in service config' > /opt/course/18/reason.txt
```

</details>

<details>
<summary>
Create Secret and mount into Pod
</summary>

```shell
$ kubectl config use-context k8s-c3-CCC
Switched to context "k8s-c3-CCC".
$ sed -i 's/todo/secret/g' /opt/course/19/secret1.yaml
$ kubectl run -n secret secret-pod --image busybox:1.31.1 --dry-run=client -oyaml > tmp.yaml
$ kubectl apply -f /opt/course/19/secret1.yaml
secret/secret1 created
$ kubectl create -n secret secret generic secret2 --from-literal user=user1 --from-literal pass=1234
secret/secret2 created
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
pod/secret-pod created
k get -n secret po secret-pod
NAME         READY   STATUS    RESTARTS   AGE
secret-pod   1/1     Running   0          18s
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: secret-pod
  name: secret-pod
  namespace: secret
spec:
  containers:
    - image: busybox:1.31.1
      name: secret-pod
      command:
        - sh
        - -c
        - "sleep 1d"
      env:
        - name: APP_USER
          valueFrom:
            secretKeyRef:
              name: secret2
              key: user
        - name: APP_PASS
          valueFrom:
            secretKeyRef:
              name: secret2
              key: pass
      volumeMounts:
        - name: secret-volume
          mountPath: /tmp/secret1
          readOnly: true
  volumes:
    - name: secret-volume
      secret:
        secretName: secret1
```

```yaml /opt/course/19/secret1.yaml
apiVersion: v1
data:
  halt: IyEgL2Jpbi9zaAojIyMgQkVHSU4gSU5JVCBJTkZPCiMgUHJvdmlkZXM6ICAgICAgICAgIGhhbHQKIyBSZXF1aXJlZC1TdGFydDoKIyBSZXF1aXJlZC1TdG9wOgojIERlZmF1bHQtU3RhcnQ6CiMgRGVmYXVsdC1TdG9wOiAgICAgIDAKIyBTaG9ydC1EZXNjcmlwdGlvbjogRXhlY3V0ZSB0aGUgaGFsdCBjb21tYW5kLgojIERlc2NyaXB0aW9uOgojIyMgRU5EIElOSVQgSU5GTwoKTkVURE9XTj15ZXMKClBBVEg9L3NiaW46L3Vzci9zYmluOi9iaW46L3Vzci9iaW4KWyAtZiAvZXRjL2RlZmF1bHQvaGFsdCBdICYmIC4gL2V0Yy9kZWZhdWx0L2hhbHQKCi4gL2xpYi9sc2IvaW5pdC1mdW5jdGlvbnMKCmRvX3N0b3AgKCkgewoJaWYgWyAiJElOSVRfSEFMVCIgPSAiIiBdCgl0aGVuCgkJY2FzZSAiJEhBTFQiIGluCgkJICBbUHBdKikKCQkJSU5JVF9IQUxUPVBPV0VST0ZGCgkJCTs7CgkJICBbSGhdKikKCQkJSU5JVF9IQUxUPUhBTFQKCQkJOzsKCQkgICopCgkJCUlOSVRfSEFMVD1QT1dFUk9GRgoJCQk7OwoJCWVzYWMKCWZpCgoJIyBTZWUgaWYgd2UgbmVlZCB0byBjdXQgdGhlIHBvd2VyLgoJaWYgWyAiJElOSVRfSEFMVCIgPSAiUE9XRVJPRkYiIF0gJiYgWyAteCAvZXRjL2luaXQuZC91cHMtbW9uaXRvciBdCgl0aGVuCgkJL2V0Yy9pbml0LmQvdXBzLW1vbml0b3IgcG93ZXJvZmYKCWZpCgoJIyBEb24ndCBzaHV0IGRvd24gZHJpdmVzIGlmIHdlJ3JlIHVzaW5nIFJBSUQuCgloZGRvd249Ii1oIgoJaWYgZ3JlcCAtcXMgJ15tZC4qYWN0aXZlJyAvcHJvYy9tZHN0YXQKCXRoZW4KCQloZGRvd249IiIKCWZpCgoJIyBJZiBJTklUX0hBTFQ9SEFMVCBkb24ndCBwb3dlcm9mZi4KCXBvd2Vyb2ZmPSItcCIKCWlmIFsgIiRJTklUX0hBTFQiID0gIkhBTFQiIF0KCXRoZW4KCQlwb3dlcm9mZj0iIgoJZmkKCgkjIE1ha2UgaXQgcG9zc2libGUgdG8gbm90IHNodXQgZG93biBuZXR3b3JrIGludGVyZmFjZXMsCgkjIG5lZWRlZCB0byB1c2Ugd2FrZS1vbi1sYW4KCW5ldGRvd249Ii1pIgoJaWYgWyAiJE5FVERPV04iID0gIm5vIiBdOyB0aGVuCgkJbmV0ZG93bj0iIgoJZmkKCglsb2dfYWN0aW9uX21zZyAiV2lsbCBub3cgaGFsdCIKCWhhbHQgLWQgLWYgJG5ldGRvd24gJHBvd2Vyb2ZmICRoZGRvd24KfQoKY2FzZSAiJDEiIGluCiAgc3RhcnR8c3RhdHVzKQoJIyBOby1vcAoJOzsKICByZXN0YXJ0fHJlbG9hZHxmb3JjZS1yZWxvYWQpCgllY2hvICJFcnJvcjogYXJndW1lbnQgJyQxJyBub3Qgc3VwcG9ydGVkIiA+JjIKCWV4aXQgMwoJOzsKICBzdG9wKQoJZG9fc3RvcAoJOzsKICAqKQoJZWNobyAiVXNhZ2U6ICQwIHN0YXJ0fHN0b3AiID4mMgoJZXhpdCAzCgk7Owplc2FjCgo6Cg==
kind: Secret
metadata:
  creationTimestamp: null
  name: secret1
  namespace: secret
```

</details>

<details>
<summary>
Update Kubernetes Version and join cluster
</summary>

```shell
$ kubectl config use-context k8s-c3-CCC
Switched to context "k8s-c3-CCC".
$ ssh cluster3-node2
$ kubectl version
Client Version: v1.28.5
...
$ apt update
$ apt-cache madison kubectl | grep 1.29.0
   kubectl | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
$ apt install kubectl=1.29.0-1.1 kubelet=1.29.0-1.1
$ kubeadm upgrade node
...
[upgrade] The configuration for this node was successfully updated!
[upgrade] Now you should go ahead and upgrade the kubelet package using your package manager.
$ systemctl daemon-reload
$ systemctl restart kubelet
$ ssh cluster3-controlplane1
$ kubeadm token create --print-join-command
kubeadm join 192.168.100.31:6443 --token u9arha.i7jyn3g1tg263t8k --discovery-token-ca-cert-hash sha256:eae975465f73f316f322bcdd5eb6a5a53f08662ecb407586561cdc06f74bf7b2
$ ssh cluster3-node2
$ kubeadm join 192.168.100.31:6443 --token u9arha.i7jyn3g1tg263t8k --discovery-token-ca-cert-hash sha256:eae975465f73f316f322bcdd5eb6a5a53f08662ecb407586561cdc06f74bf7b2
...
This node has joined the cluster:
* Certificate signing request was sent to apiserver and a response was received.
* The Kubelet was informed of the new secure connection details.
...
$ kubectl get node
NAME                     STATUS   ROLES           AGE     VERSION
cluster3-controlplane1   Ready    control-plane   120d    v1.29.0
cluster3-node1           Ready    <none>          120d    v1.29.0
cluster3-node2           Ready    <none>          8m41s   v1.29.0
```

</details>

<details>
<summary>
Create a Static Pod and Service
</summary>

```shell
$ kubectl config use-context k8s-c3-CCC
Switched to context "k8s-c3-CCC".
$ ssh cluster3-controlplane1
$ kubectl run my-static-pod --image nginx:1.16-alpine --dry-run=client -oyaml > tmp.yaml
$ mv tmp.yaml /etc/kubernetes/manifests/
$ kubectl get po my-static-pod-cluster3-controlplane1
NAME                                   READY   STATUS    RESTARTS   AGE
my-static-pod-cluster3-controlplane1   1/1     Running   0          13s
$ kubectl expose po my-static-pod-cluster3-controlplane1 --name static-pod-service --type NodePort --port 80
service/static-pod-service exposed
$ kubectl get all -l run=my-static-pod
NAME                                       READY   STATUS    RESTARTS   AGE
pod/my-static-pod-cluster3-controlplane1   1/1     Running   0          2m17s

NAME                         TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
service/static-pod-service   NodePort   10.107.96.74   <none>        80:31279/TCP   81s
```

```yaml tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: my-static-pod
  name: my-static-pod
spec:
  containers:
  - image: nginx:1.16-alpine
    name: my-static-pod
    resources:
      requests:
        memory: 20Mi
        cpu: 10m
```

</details>

<details>
<summary>
Check how long certificates are valid
</summary>

```shell
$ kubectl config use-context k8s-c2-AC
Switched to context "k8s-c2-AC".
$ ssh cluster2-controlplane1
$ openssl x509 -in /etc/kubernetes/pki/apiserver.crt -text -noout| grep Valid -A2
        Validity
            Not Before: Jan  4 10:13:37 2024 GMT
            Not After : May  3 11:17:25 2025 GMT
$ echo 'May  3 11:17:25 2025 GMT' > /opt/course/22/expiration
$ echo 'kubeadm certs renew apiserver' > /opt/course/22/kubeadm-renew-certs.sh
```

</details>

<details>
<summary>
kubelet client/server cert info
</summary>

```shell
$ kubectl config use-context k8s-c2-AC
Switched to context "k8s-c2-AC".
$ ssh cluster2-node1
$ openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -text -noout | grep Issuer
        Issuer: CN = kubernetes
$ openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -text -noout | grep 'Extended Key Usage' -A1
            X509v3 Extended Key Usage:
                TLS Web Client Authentication
$ openssl x509 -in /var/lib/kubelet/pki/kubelet.crt -text -noout | grep Issuer
        Issuer: CN = cluster2-node1-ca@1704363800
$ openssl x509 -in /var/lib/kubelet/pki/kubelet.crt -text -noout | grep 'Extended Key Usage' -A1
            X509v3 Extended Key Usage:
                TLS Web Server Authentication
$ vi /opt/course/23/certificate-info.txt
```

</details>

<details>
<summary>
NetworkPolicy
</summary>

```shell
$ kubectl config use-context k8s-c1-H
Switched to context "k8s-c1-H".
$ kubectl get -n project-snake po -L app
NAME        READY   STATUS    RESTARTS   AGE    APP
backend-0   1/1     Running   0          120d   backend
db1-0       1/1     Running   0          120d   db1
db2-0       1/1     Running   0          120d   db2
vault-0     1/1     Running   0          120d   vault
$ kubectl get -n project-snake po -owide
NAME        READY   STATUS    RESTARTS   AGE    IP           NODE             NOMINATED NODE   READINESS GATES
backend-0   1/1     Running   0          120d   10.44.0.10   cluster1-node1   <none>           <none>
db1-0       1/1     Running   0          120d   10.36.0.16   cluster1-node2   <none>           <none>
db2-0       1/1     Running   0          120d   10.36.0.12   cluster1-node2   <none>           <none>
vault-0     1/1     Running   0          120d   10.44.0.19   cluster1-node1   <none>           <none>
$ kubectl exec -n project-snake backend-0 -- curl -s 10.36.0.16:1111
database one
$ kubectl exec -n project-snake backend-0 -- curl -s 10.36.0.12:2222
database two
$ kubectl exec -n project-snake backend-0 -- curl -s 10.44.0.19:3333
vault secret storage
$ vi tmp.yaml
$ kubectl apply -f tmp.yaml
networkpolicy.networking.k8s.io/np-backend created
$ kubectl exec -n project-snake backend-0 -- curl -s 10.36.0.16:1111
database one
$ kubectl exec -n project-snake backend-0 -- curl -s 10.36.0.12:2222
database two
$ kubectl exec -n project-snake backend-0 -- curl -s 10.44.0.19:3333
command terminated with exit code 28
```

```yaml tmp.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: np-backend
  namespace: project-snake
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: db1
      ports:
        - protocol: TCP
          port: 1111
    - to:
        - podSelector:
            matchLabels:
              app: db2
      ports:
        - protocol: TCP
          port: 2222
```

</details>

<details>
<summary>
etcd Snapshot Save and Restore
</summary>

```shell
$ kubectl config use-context k8s-c3-CCC
Switched to context "k8s-c3-CCC".
$ ssh cluster3-controlplane1
$ kubectl describe -n kube-system po etcd-cluster3-controlplane1 | grep ca
...
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
$ ca=/etc/kubernetes/pki/etcd/ca.crt
$ kubectl describe -n kube-system po etcd-cluster3-controlplane1 | grep cert
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
...
$ cert=/etc/kubernetes/pki/etcd/server.crt
$ kubectl describe -n kube-system po etcd-cluster3-controlplane1 | grep key
      --key-file=/etc/kubernetes/pki/etcd/server.key
...
$ key=/etc/kubernetes/pki/etcd/server.key
$ ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
  --cacert=$ca --cert=$cert --key=$key \
  snapshot save /tmp/etcd-backup.db
$ kubectl run zerohertz --image zerohertz
pod/zerohertz created
$ kubectl get po zerohertz
NAME        READY   STATUS         RESTARTS   AGE
zerohertz   0/1     ErrImagePull   0          10s
$ cat /etc/kubernetes/manifests/etcd.yaml | grep data-dir
    - --data-dir=/var/lib/etcd
$ ETCDCTL_API=3 etcdctl --data-dir /var/lib/etcd-backup snapshot restore /tmp/etcd-backup.db
$ sed -i 's|/var/lib/etcd|/var/lib/etcd-backup|g' /etc/kubernetes/manifests/etcd.yaml
$ kubectl get po zerohertz
Error from server (NotFound): pods "zerohertz" not found
```

</details>

---

# Tips

> 빈출 유형

+ Cluster Architecture, Installation & Configuration
  + Cluster Upgrade (`kubeadm`)
  + etcd Backup & Restore (`etcdctl snapshot`)
  + RBAC (`ServiceAccount`, `Role`, `RoleBinding`)
+ Workloads & Scheduling
  + Node: Drain & Uncordon (`kubectl drain`, `kubectl uncordon`)
  + Deployment: Rolling Update & Undo Rollback
  + Deployment: Scaling (`kubectl scale`)
  + Pod: Toleration
  + Pod: Multi-container
+ Services & Networking
  + Deployment & Service (`kubectl expose`)
  + NetworkPolicy (`namespaceSelector`, `podSelector`, `ports`)
  + Ingress (`curl`)
+ Storage
  + Pod: Sidecar (`emptyDir`)
  + Pod: PV, PVC
+ Troubleshooting
  + Node: notReady (`systemctl status kubelet`)
  + Monitoring (`kubectl top`)
  + Pod: Log (`kubectl logs`)
  + `-ojsonpath` & `--sort-by`

<details>
<summary>
Shortnames
</summary>

```shell
$ kubectl api-resources
NAME                              SHORTNAMES   APIVERSION                        NAMESPACED   KIND
bindings                                       v1                                true         Binding
componentstatuses                 cs           v1                                false        ComponentStatus
configmaps                        cm           v1                                true         ConfigMap
endpoints                         ep           v1                                true         Endpoints
events                            ev           v1                                true         Event
limitranges                       limits       v1                                true         LimitRange
namespaces                        ns           v1                                false        Namespace
nodes                             no           v1                                false        Node
persistentvolumeclaims            pvc          v1                                true         PersistentVolumeClaim
persistentvolumes                 pv           v1                                false        PersistentVolume
pods                              po           v1                                true         Pod
podtemplates                                   v1                                true         PodTemplate
replicationcontrollers            rc           v1                                true         ReplicationController
resourcequotas                    quota        v1                                true         ResourceQuota
secrets                                        v1                                true         Secret
serviceaccounts                   sa           v1                                true         ServiceAccount
services                          svc          v1                                true         Service
mutatingwebhookconfigurations                  admissionregistration.k8s.io/v1   false        MutatingWebhookConfiguration
validatingwebhookconfigurations                admissionregistration.k8s.io/v1   false        ValidatingWebhookConfiguration
customresourcedefinitions         crd,crds     apiextensions.k8s.io/v1           false        CustomResourceDefinition
apiservices                                    apiregistration.k8s.io/v1         false        APIService
controllerrevisions                            apps/v1                           true         ControllerRevision
daemonsets                        ds           apps/v1                           true         DaemonSet
deployments                       deploy       apps/v1                           true         Deployment
replicasets                       rs           apps/v1                           true         ReplicaSet
statefulsets                      sts          apps/v1                           true         StatefulSet
selfsubjectreviews                             authentication.k8s.io/v1          false        SelfSubjectReview
tokenreviews                                   authentication.k8s.io/v1          false        TokenReview
localsubjectaccessreviews                      authorization.k8s.io/v1           true         LocalSubjectAccessReview
selfsubjectaccessreviews                       authorization.k8s.io/v1           false        SelfSubjectAccessReview
selfsubjectrulesreviews                        authorization.k8s.io/v1           false        SelfSubjectRulesReview
subjectaccessreviews                           authorization.k8s.io/v1           false        SubjectAccessReview
horizontalpodautoscalers          hpa          autoscaling/v2                    true         HorizontalPodAutoscaler
cronjobs                          cj           batch/v1                          true         CronJob
jobs                                           batch/v1                          true         Job
certificatesigningrequests        csr          certificates.k8s.io/v1            false        CertificateSigningRequest
leases                                         coordination.k8s.io/v1            true         Lease
endpointslices                                 discovery.k8s.io/v1               true         EndpointSlice
events                            ev           events.k8s.io/v1                  true         Event
flowschemas                                    flowcontrol.apiserver.k8s.io/v1   false        FlowSchema
prioritylevelconfigurations                    flowcontrol.apiserver.k8s.io/v1   false        PriorityLevelConfiguration
ingressclasses                                 networking.k8s.io/v1              false        IngressClass
ingresses                         ing          networking.k8s.io/v1              true         Ingress
networkpolicies                   netpol       networking.k8s.io/v1              true         NetworkPolicy
runtimeclasses                                 node.k8s.io/v1                    false        RuntimeClass
poddisruptionbudgets              pdb          policy/v1                         true         PodDisruptionBudget
clusterrolebindings                            rbac.authorization.k8s.io/v1      false        ClusterRoleBinding
clusterroles                                   rbac.authorization.k8s.io/v1      false        ClusterRole
rolebindings                                   rbac.authorization.k8s.io/v1      true         RoleBinding
roles                                          rbac.authorization.k8s.io/v1      true         Role
priorityclasses                   pc           scheduling.k8s.io/v1              false        PriorityClass
csidrivers                                     storage.k8s.io/v1                 false        CSIDriver
csinodes                                       storage.k8s.io/v1                 false        CSINode
csistoragecapacities                           storage.k8s.io/v1                 true         CSIStorageCapacity
storageclasses                    sc           storage.k8s.io/v1                 false        StorageClass
volumeattachments                              storage.k8s.io/v1                 false        VolumeAttachment
```

</details>

<details>
<summary>
OpenSSL
</summary>

> Network 통신과 data를 암호화하는 데 사용되는 도구

```shell
$ kubectl describe -n kube-system po kube-apiserver-controlplane | grep -i cert
...
      --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
...
$ openssl x509 -in /etc/kubernetes/pki/apiserver.crt -text -noout
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 8339414143901577932 (0x73bb8cfad21f32cc)
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: CN = kubernetes
        Validity
            Not Before: Apr 14 07:24:30 2024 GMT
            Not After : Apr 14 07:29:30 2025 GMT
        Subject: CN = kube-apiserver
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
...
$ kubectl describe -n kube-system po etcd-controlplane | grep -i cert
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
...
$ openssl x509 -in /etc/kubernetes/pki/etcd/server.crt -text -noout
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 4154004745008470301 (0x39a5fd4e8584591d)
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: CN = etcd-ca
        Validity
            Not Before: Apr 14 07:24:31 2024 GMT
            Not After : Apr 14 07:29:31 2025 GMT
        Subject: CN = controlplane
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
```

+ `x509`: [X.509](https://ko.wikipedia.org/wiki/X.509) (공개 키 국제 표준 인증서) 설정
+ `-in` ${FILE}: 입력 파일 지정
+ `-text`: 인증서의 내용을 text 형식으로 출력
+ `-noout`: 출력에서 인증서의 encoding 원본 data 제외

</details>
