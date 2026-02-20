---
title: CKAD (Certified Kubernetes Application Developer)
date: 2026-02-20 14:07:12
categories:
  - 3. DevOps
tags:
  - Docker
  - Kubernetes
---

# Introduction

> CKAD: Kubernetes 환경에서 cloud-native application을 효과적으로 설계, 구축, 배포 및 구성할 수 있는 역량을 검증하는 CNCF의 공식 자격증

<img src="/images/ckad/certificate.webp" alt="certificate" width=700 />

[CKAD curriculum](https://github.com/cncf/curriculum/blob/0231af9c8004b20254da4d50cd3c8b898e91e86d/CKAD_Curriculum_v1.34.pdf)에서 CKAD가 포함하는 내용들을 아래와 같이 확인할 수 있다.

<div align="right"><code>v1.34</code> 기준</div>

|                          Domain                          | Weight | Key Points                                                                                                                                                                                                                                                                                                                                                                                              |
| :------------------------------------------------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|               Application Design and Build               |  20%   | ✅ Define, build and modify container images<br/>✅ Choose and use the right workload resource (Deployment, DaemonSet, CronJob, etc.)<br/>✅ Understand multi-container Pod design patterns (e.g. sidecar, init and others)<br/>✅ Utilize persistent and ephemeral volumes                                                                                                                             |
|                  Application Deployment                  |  20%   | ✅ Use Kubernetes primitives to implement common deployment strategies (e.g. blue/green or canary)<br/>✅ Understand Deployments and how to perform rolling updates<br/>✅ Use the Helm package manager to deploy existing packages<br/>✅ Kustomize                                                                                                                                                    |
|      Application Observability<br/> and Maintenance      |  15%   | ✅ Understand API depreciations<br/>✅ Implement probes and health checks<br/>✅ Use built-in CLI tools to monitor Kubernetes applications<br/>✅ Utilize container logs<br/>✅ Debugging in Kubernetes                                                                                                                                                                                                 |
| Application Environment,<br/> Configuration and Security |  25%   | ✅ Discover and use resources that extend Kubernetes (CRD, Operators)<br/>✅ Understand authentication, authorization and admission control<br/>✅ Understand requests, limits, quotas<br/>✅ Define resource requirements<br/>✅ Understand ConfigMaps<br/>✅ Create & consume Secrets<br/>✅ Understand ServiceAccounts<br/>✅ Understand Application Security (SecurityContexts, Capabilities, etc.) |
|                 Services and Networking                  |  20%   | ✅ Demonstrate basic understanding of NetworkPolicies<br/>✅ Provide and troubleshoot access to applications via services<br/>✅ Use Ingress rules to expose applications                                                                                                                                                                                                                               |

<!-- More -->

- 가격: \\$395 (현재는 \\$445)
- 시간: 2시간
- 문제: 15-20문제
- 장소: 사방이 막힌 조용한 장소
- 준비물: 신분증 (영문 이름 필수)

[공식 사이트](https://training.linuxfoundation.org/certification/certified-kubernetes-application-developer-ckad/)에서 결제하여 CKAD 응시를 신청할 수 있다.

<img src="/images/ckad/coupon.png" alt="coupon" width="500" />

위와 같이 Linux Foundation에서 제공하는 coupon을 통해 기존 \\$395의 가격을 할인 받을 수 있다. (필자는 50%의 할인을 받아 \\$197.5에 결제했다.)
결제를 마쳤다면 1년 내로 아래와 같이 시험을 예약해야 한다.

<img src="/images/ckad/ckad-exam-date.png" alt="ckad-exam-date" width="500" />

[CKA](/cka/)와 동일하게, [시험 응시 시 환경](https://docs.linuxfoundation.org/tc-docs/certification/tips-cka-and-ckad#exam-technical-instructions)에서는 현재 존재하지 않지만 multi-cluster 환경에서 시험을 응시하고, [여기](https://test-takers.psiexams.com/linux/manage)에서 시험 응시 시 사용할 기기의 검증을 수행할 수 있다.
또한 CKAD도 [Udemy에서 Mumshad님이 진행하신 강의](https://www.udemy.com/course/certified-kubernetes-application-developer)가 매우 유명하기 때문에 해당 강의를 수강했다.
해당 강의를 수강하면 [KodeKloud](https://learn.kodekloud.com/user/courses/udemy-labs-certified-kubernetes-application-developer)를 통해 실제 시험과 유사한 조건 속에서 연습할 수 있다.
마지막으로 CKA 시험의 결제를 마치면 아래와 같이 [killer.sh](https://killer.sh/)의 문제를 2회 풀 수 있는 권한을 주기 때문에 복기를 위해 이를 풀었다.

<img src="/images/ckad/killer.sh.png" alt="killer.sh" width="500" />

---

# Theoretical Backgrounds

## Recap Core Concepts

<details>
<summary>
Pods
</summary>

```shell
$ kubectl get po | wc -l
No resources found in default namespace.
0
$ kubectl run nginx --image=nginx
pod/nginx created
$ kubectl get po | wc -l
5
$ kubectl describe po newpods- | grep Image:
    Image:         busybox
    Image:         busybox
    Image:         busybox
$ kubectl get po -owide
NAME            READY   STATUS    RESTARTS   AGE    IP           NODE           NOMINATED NODE   READINESS GATES
newpods-2f7rm   1/1     Running   0          108s   10.22.0.11   controlplane   <none>           <none>
newpods-2r9kx   1/1     Running   0          108s   10.22.0.9    controlplane   <none>           <none>
newpods-srqvs   1/1     Running   0          108s   10.22.0.10   controlplane   <none>           <none>
po              1/1     Running   0          89s    10.22.0.12   controlplane   <none>           <none>
$ kubectl get po webapp
NAME     READY   STATUS         RESTARTS   AGE
webapp   1/2     ErrImagePull   0          16s
$ kubectl describe po webapp | grep Image:
    Image:          nginx
    Image:          agentx
$ kubectl describe po webapp | grep State: -A 1
    State:          Running
      Started:      Mon, 10 Nov 2025 03:49:05 +0000
--
    State:          Waiting
      Reason:       ImagePullBackOff
$ kubectl get event | grep webapp | grep Error
10m         Warning   Failed                           pod/webapp           Error: ErrImagePull
3m5s        Warning   Failed                           pod/webapp           Error: ImagePullBackOff
$ kubectl delete po webapp
pod "webapp" deleted from default namespace
$ kubectl run redis --image=redis123
pod/redis created
$ kubectl edit po redis
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
$ kubectl get rs
NAME              DESIRED   CURRENT   READY   AGE
new-replica-set   4         4         0       59s
$ kubectl describe po | grep Image:
    Image:         busybox777
    Image:         busybox777
    Image:         busybox777
    Image:         busybox777
$ kubectl get po
NAME                    READY   STATUS             RESTARTS   AGE
new-replica-set-6g9wk   0/1     ErrImagePull       0          109s
new-replica-set-76ggt   0/1     ErrImagePull       0          109s
new-replica-set-94z4s   0/1     ImagePullBackOff   0          109s
new-replica-set-pr5jx   0/1     ImagePullBackOff   0          109s
$ kubectl delete po new-replica-set-6g9wk
pod "new-replica-set-6g9wk" deleted from default namespace
$ sed -i "s|v1|apps/v1|g" replicaset-definition-1.yaml
$ kubectl apply -f replicaset-definition-1.yaml
replicaset.apps/replicaset-1 created
$ sed -i "s|frontend|nginx|g" replicaset-definition-2.yaml
$ kubectl apply -f replicaset-definition-2.yaml
replicaset.apps/replicaset-2 created
$ kubectl delete rs replicaset-1 replicaset-2
replicaset.apps "replicaset-1" deleted from default namespace
replicaset.apps "replicaset-2" deleted from default namespace
$ sed -i "s|busybox777|busybox|g" new-replica-set.yaml
$ kubectl apply -f new-replica-set.yaml
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
$ kubectl get po | wc -l
No resources found in default namespace.
0
$ kubectl get rs | wc -l
No resources found in default namespace.
0
$ kubectl get deploy | wc -l
No resources found in default namespace.
0
$ kubectl get deploy | wc -l
2
$ kubectl get rs | wc -l
2
$ kubectl get po | wc -l
5
$ kubectl get po
NAME                                   READY   STATUS             RESTARTS   AGE
frontend-deployment-557f78f544-dkkkv   0/1     ImagePullBackOff   0          96s
frontend-deployment-557f78f544-kvwxp   0/1     ImagePullBackOff   0          96s
frontend-deployment-557f78f544-pgcs4   0/1     ImagePullBackOff   0          96s
frontend-deployment-557f78f544-x6h2m   0/1     ImagePullBackOff   0          96s
$ kubectl describe po | grep Image:
    Image:         busybox888
    Image:         busybox888
    Image:         busybox888
    Image:         busybox888
$ sed -i 's|kind: deployment|kind: Deployment|g' deployment-definition-1.yaml
$ kubectl apply -f deployment-definition-1.yaml
deployment.apps/deployment-1 created
$ kubectl create deploy httpd-frontend --image httpd:2.4-alpine --replicas 3
deployment.apps/httpd-frontend created
```

</details>

<details>
<summary>
Namespaces
</summary>

```shell
$ kubectl get ns | wc -l
11
$ kubectl get po -n research | wc -l
3
$ kubectl run -n finance redis --image redis
pod/redis created
$ kubectl get po -A | grep blue
marketing       blue                                      1/1     Running            0             2m52s
$ kubectl get svc -A | grep redis
dev             redis-db-service   ClusterIP      10.43.113.176   <none>           6379/TCP                     3m59s
marketing       redis-db-service   NodePort       10.43.145.241   <none>           6379:31041/TCP               3m59s
```

</details>

<details>
<summary>
Imperative Commands
</summary>

```shell
$ kubectl run nginx-pod --image nginx:alpine
pod/nginx-pod created
$ kubectl run redis --image redis:alpine --labels tier=db
pod/redis created
$ kubectl expose po redis --name redis-service --port 6379
service/redis-service exposed
$ kubectl create deploy webapp --image kodekloud/webapp-color --replicas 3
deployment.apps/webapp created
$ kubectl run custom-nginx --image nginx --dry-run=client -oyaml > tmp.yaml
$ vim tmp.yaml
...
    ports:
      - containerPort: 8080
...
$ kubectl apply -f tmp.yaml
pod/custom-nginx created
$ kubectl create ns dev-ns
namespace/dev-ns created
$ kubectl create deploy redis-deploy -n dev-ns --image redis --replicas 2
deployment.apps/redis-deploy created
$ kubectl run httpd --image httpd:alpine
pod/httpd created
$ kubectl expose po httpd --port 80
service/httpd exposed
$ kubectl explain pod
KIND:       Pod
VERSION:    v1

DESCRIPTION:
    Pod is a collection of containers that can run on a host. This resource is
    created by clients and scheduled onto hosts.
...
$ kubectl explain pod.spec.containers
KIND:       Pod
VERSION:    v1

FIELD: containers <[]Container>


DESCRIPTION:
    List of containers belonging to the pod. Containers cannot currently be
    added or removed. There must be at least one container in a Pod. Cannot be
    updated.
    A single application container that you want to run within a pod.
...
$ kubectl explain deploy.spec.replicas
GROUP:      apps
KIND:       Deployment
VERSION:    v1

FIELD: replicas <integer>


DESCRIPTION:
    Number of desired pods. This is a pointer to distinguish between explicit
    zero and not specified. Defaults to 1.
$ kubectl explain service.spec.ports --recursive
KIND:       Service
VERSION:    v1

FIELD: ports <[]ServicePort>


DESCRIPTION:
    The list of ports that are exposed by this service. More info:
    https://kubernetes.io/docs/concepts/services-networking/service/#virtual-ips-and-service-proxies
    ServicePort contains information on service's port.

FIELDS:
  appProtocol   <string>
  name  <string>
  nodePort      <integer>
  port  <integer> -required-
  protocol      <string>
  enum: SCTP, TCP, UDP
  targetPort    <IntOrString>
```

</details>

## Configuration

<details>
<summary>
Commands and Arguments
</summary>

```shell
$ kubectl get po | wc -l
2
$ kubectl get po -oyaml | grep command -A2
    - command:
      - sleep
      - "4800"
$ vim ubuntu-sleeper-2.yaml
...
    command:
      - sleep
      - "5000"
...
$ kubectl apply -f ubuntu-sleeper-2.yaml
pod/ubuntu-sleeper-2 created
$ sed -i "s|1200|'1200'|g" ubuntu-sleeper-3.yaml
$ kubectl apply -f ubuntu-sleeper-3.yaml
pod/ubuntu-sleeper-3 created
$ sed -i "s|'1200'|'2000'|g" ubuntu-sleeper-3.yaml
$ kubectl delete -f ubuntu-sleeper-3.yaml --force
Warning: Immediate deletion does not wait for confirmation that the running resource has been terminated. The resource may continue to run on the cluster indefinitely.
pod "ubuntu-sleeper-3" force deleted from default namespace
$ kubectl apply -f ubuntu-sleeper-3.yaml
pod/ubuntu-sleeper-3 created
$ cat webapp-color/Dockerfile | tail -n 1
ENTRYPOINT ["python", "app.py"]
$ cat webapp-color/Dockerfile2 | tail -n 3
ENTRYPOINT ["python", "app.py"]

CMD ["--color", "red"]
$ cat webapp-color-2/Dockerfile | tail -n 3
ENTRYPOINT ["python", "app.py"]

CMD ["--color", "red"]
$ cat webapp-color-2/webapp-color-pod.yaml | tail -n 1
    command: ["--color","green"]
$ cat webapp-color-3/Dockerfile | tail -n 3
ENTRYPOINT ["python", "app.py"]

CMD ["--color", "red"]
$ cat webapp-color-3/webapp-color-pod-2.yaml | tail -n 2
    command: ["python", "app.py"]
    args: ["--color", "pink"]
$ kubectl run webapp-green --image kodekloud/webapp-color -- '--color=green'
pod/webapp-green created
```

</details>

<details>
<summary>
ConfigMaps
</summary>

```shell
$ kubectl get po | wc -l
2
$ kubectl describe po  | grep -i env -A 1
    Environment:
      APP_COLOR:  pink
$ kubectl get po webapp-color -oyaml > tmp.yaml
$ sed -i "s|pink|green|g" tmp.yaml
$ kubectl delete -f tmp.yaml
pod "webapp-color" deleted from default namespace
$ kubectl apply -f tmp.yaml
pod/webapp-color created
$ kubectl get cm | wc -l
3
$ kubectl get cm db-config -oyaml | grep -i host
  DB_HOST: SQL01.example.com
$ kubectl create cm webapp-config-map --from-literal=APP_COLOR=darkblue --from-literal=APP_OTHER=disregard
configmap/webapp-config-map created
$ kubectl get po webapp-color -oyaml > tmp.yaml
$ kubectl delete -f tmp.yaml
pod "webapp-color" deleted from default namespace
$ vim tmp.yaml
...
  - env:
    - name: APP_COLOR
      valueFrom:
        configMapKeyRef:
          name: webapp-config-map
          key: APP_COLOR
...
$ kubectl apply -f tmp.yaml
pod/webapp-color created
```

</details>

<details>
<summary>
Secrets
</summary>

```shell
$ kubectl get secret | wc -l
2
$ kubectl describe secret dashboard-token | grep Data -A100
Data
====
ca.crt:     566 bytes
namespace:  7 bytes
token:      eyJhbGciOiJSUzI1NiIsImtpZCI6Ik94T09pd3dUemdWbnozbjFKM1pCcFJuYllSY3ZWY3hudTBFZTdYM0hacFEifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImRhc2hib2FyZC10b2tlbiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJkYXNoYm9hcmQtc2EiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiJlOGQyOTRkMy05YzFiLTQ0Y2UtOTYwZS1iMjdmZDRiOWY5ZjQiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpkYXNoYm9hcmQtc2EifQ.lp-wlFCEAWPU9X7nL2l2aeaPZyisIQSx3H9zmf5UcHUr1AwyFEObzTL660DCnyS6xgfE_-g7_bnCzqfPpl31vWmNvpE8_H9FRVe4-rMe5ZFkju1KbGECuP05ek-kmWF3ZSF_YAJwR5eBoim1BimK3zCyRZoe06ebHUDbIn-Lx8v3Pitz7Nw_JPQKWSVKdZWqiBxtD9e9GRPhwt_7KSsh5dNh9Z3SbhO2y-wBOnTbJB4Cz4iPszEE49FjKAm4kxinB4ue95PUYsryXeuCgSwq8nesJ4cvsJzi6b-jsK-vnnQvSv02Ot9CmERztkSZMraoKjrorPPVj40NwnBRVM-fDQ
$ kubectl describe secret dashboard-token | grep -i type
Type:  kubernetes.io/service-account-token
$ kubectl create secret generic db-secret --from-literal=DB_Host=sql01 --from-literal=DB_User=root --from-literal=DB_Password=password123
secret/db-secret created
$ kubectl get po webapp-pod -oyaml > tmp.yaml
$ vim tmp.yaml
...
    envFrom:
    - secretRef:
        name: db-secret
...
$ kubectl delete -f tmp.yaml
pod "webapp-pod" deleted from default namespace
$ kubectl apply -f tmp.yaml
pod/webapp-pod created
```

</details>

<details>
<summary>
Security Contexts
</summary>

```shell
$ kubectl exec -it ubuntu-sleeper -- whoami
root
$ kubectl get po ubuntu-sleeper -oyaml > tmp.yaml
$ vim tmp.yaml
...
  securityContext:
    runAsUser: 1010
...
$ kubectl delete -f tmp.yaml
pod "ubuntu-sleeper" deleted from default namespace
$ kubectl apply -f tmp.yaml
pod/ubuntu-sleeper created
$ cat multi-pod.yaml | grep security -A1
  securityContext:
    runAsUser: 1001
--
     securityContext:
      runAsUser: 1002
$ vim tmp.yaml
...
    securityContext:
      runAsUser: 0
      capabilities:
        add: ["SYS_TIME"]
...
$ kubectl delete -f tmp.yaml
pod "ubuntu-sleeper" deleted from default namespace
$ kubectl apply -f tmp.yaml
pod/ubuntu-sleeper created
$ vim tmp.yaml
...
    securityContext:
      runAsUser: 0
      capabilities:
        add: ["SYS_TIME", "NET_ADMIN"]
...
$ kubectl delete -f tmp.yaml
pod "ubuntu-sleeper" deleted from default namespace
$ kubectl apply -f tmp.yaml
pod/ubuntu-sleeper created
```

</details>

<details>
<summary>
Resource Limits
</summary>

```shell
$ kubectl describe po rabbit | grep cpu: -B1
    Limits:
      cpu:  1
    Requests:
      cpu:        500m
$ kubectl delete po rabbit
pod "rabbit" deleted from default namespace
$ kubectl describe po elephant | grep Reason: -A1
      Reason:       CrashLoopBackOff
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    137
$ kubectl get po elephant -oyaml | sed 's/10Mi/20Mi/' | kubectl replace --force -f -
pod "elephant" deleted from default namespace
pod/elephant replaced
$ kubectl delete po elephant
pod "elephant" deleted from default namespace
```

</details>

<details>
<summary>
Service Account
</summary>

```shell
$ kubectl get sa | wc -l
3
$ kubectl describe deploy web-dashboard | grep Image:
    Image:      gcr.io/kodekloud/customimage/my-kubernetes-dashboard
$ kubectl get pods -oyaml | grep serviceAccountName
    serviceAccountName: default
$ kubectl describe po web-dashboard-7666579d69-n6tvn | grep Mounts: -A1
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-8rxh4 (ro)
$ kubectl create sa dashboard-sa
serviceaccount/dashboard-sa created
$ kubectl create token dashboard-sa
eyJhbGciOiJSUzI1NiIsImtpZCI6InEtZTFQYmlsT3J0MjV1aWljeWlFcUVuTkxrcE9JXzR4SVlSaHA5WmdDSFkifQ.eyJhdWQiOlsiaHR0cHM6Ly9rdWJlcm5ldGVzLmRlZmF1bHQuc3ZjLmNsdXN0ZXIubG9jYWwiLCJrM3MiXSwiZXhwIjoxNzcxMzA5NDYwLCJpYXQiOjE3NzEzMDU4NjAsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiNThmYjIxZTItNmM0MS00NjA0LTkwYTEtZDA0MzQ3NjJiMDkyIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJkZWZhdWx0Iiwic2VydmljZWFjY291bnQiOnsibmFtZSI6ImRhc2hib2FyZC1zYSIsInVpZCI6IjVmNTVkY2VjLWNhMWUtNGFhYy05YTAyLWY1ZDZlY2U2OTE4NyJ9fSwibmJmIjoxNzcxMzA1ODYwLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpkYXNoYm9hcmQtc2EifQ.eRXGCFIPmDIe1O7RpKyHTcJd9Ub2L4yPE7IvFAtQEj95IMaOLIFRoskRrsZ3MsPFNsUCewXqXhq-1u7fEw40wvBa3zNJ-dDoPxv3GC9uO951OJWruxz47Ab31B5oct5hW0jfArPNZ7Pzjm92uSKCkpT4gw7J5jNb_t_VhsouXuk7A7VHm5_-YhXVrLq_uSrhSeGhpOHvwuOkQ5xLigiW7rg473N1aXlop38A91ltrfHwoR18GVmGy5HQ7RGnuF9s0j3f_c8VNCIUujwlUY7hSUTVTvccoI1dQFp4knNch1KAhenVwmdpFVTHVrHuKyi1SqHRXVic7kWBeLkWXhn9Zg
$ kubectl get deploy web-dashboard -oyaml > tmp.yaml
$ vim tmp.yaml
...
    spec:
      serviceAccountName: dashboard-sa
...
$ kubectl apply -f tmp.yaml
deployment.apps/web-dashboard configured
$ kubectl get sa dashboard-sa -oyaml | sed '$a automountServiceAccountToken: false' | kubectl apply -f -
serviceaccount/dashboard-sa configured
$ vim tmp.yaml
...
        volumeMounts:
        - name: token
          mountPath: "/var/run/secrets/kubernetes.io/serviceaccount"
          readOnly: true
...
      volumes:
      - name: token
        projected:
          sources:
          - serviceAccountToken:
              path: token
...
$ kubectl replace --force -f tmp.yaml
deployment.apps "web-dashboard" deleted from default namespace
deployment.apps/web-dashboard replaced
$ kubectl exec $(kubectl get pod -l name=web-dashboard -o jsonpath='{.items[0].metadata.name}') -- ls /var/run/secrets/kubernetes.io/serviceaccount/
token
```

</details>

<details>
<summary>
Taints and Tolerations
</summary>

```shell
$ kubectl get node | wc -l
3
$ kubectl describe node node01 | grep -i taint
Taints:             <none>
$ kubectl taint node node01 spray=mortein:NoSchedule
node/node01 tainted
$ kubectl run mosquito --image=nginx
pod/mosquito created
$ kubectl get po
NAME       READY   STATUS    RESTARTS   AGE
mosquito   0/1     Pending   0          17s
$ kubectl describe po mosquito | grep -i event -A10
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  55s   default-scheduler  0/2 nodes are available: 1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }, 1 node(s) had untolerated taint {spray: mortein}. no new claims to deallocate, preemption: 0/2 nodes are available: 2 Preemption is not helpful for scheduling.
$ kubectl run bee --image=nginx --dry-run=client -oyaml | sed '/^spec:/a\  tolerations:\n\  - key: spray\n\    value: mortein\n    effect: NoSchedule' | kubectl apply -f -
pod/bee created
$ kubectl describe node controlplane | grep -i taint
Taints:             node-role.kubernetes.io/control-plane:NoSchedule
$ kubectl taint node controlplane node-role.kubernetes.io/control-plane:NoSchedule-
node/controlplane untainted
$ kubectl get po mosquito -owide
NAME       READY   STATUS    RESTARTS   AGE   IP           NODE           NOMINATED NODE   READINESS GATES
mosquito   1/1     Running   0          12m   172.17.0.4   controlplane   <none>           <none>
```

</details>

<details>
<summary>
Node Affinity
</summary>

```shell
$ kubectl describe node node01 | grep -i label -A5
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=node01
                    kubernetes.io/os=linux
Annotations:        flannel.alpha.coreos.com/backend-data: {"VNI":1,"VtepMAC":"96:70:53:4c:6c:15"}
$ kubectl label node node01 color=blue
node/node01 labeled
$ kubectl create deploy blue --image nginx --replicas 3
deployment.apps/blue created
$ kubectl describe node | grep -i taint
Taints:             <none>
Taints:             <none>
$ kubectl edit deploy blue
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
NAME                    READY   STATUS    RESTARTS   AGE   IP           NODE     NOMINATED NODE   READINESS GATES
blue-56f4cbb9f6-4dxdz   1/1     Running   0          27s   172.17.1.5   node01   <none>           <none>
blue-56f4cbb9f6-7qjdc   1/1     Running   0          29s   172.17.1.4   node01   <none>           <none>
blue-56f4cbb9f6-m4ddl   1/1     Running   0          25s   172.17.1.6   node01   <none>           <none>
$ kubectl get deploy blue -oyaml | sed 's/replicas: 3/replicas: 2/' | sed 's/blue/red/g' | sed 's|key: color|key: node-role.kubernetes.io/control-plane|' | sed 's/operator: In/operator: Exists/' | sed -z 's/values:\n.*- red//' | kubectl apply -f -
deployment.apps/red created
```

</details>

## Multi-Container PODs

<details>
<summary>
Multi-Container PODs
</summary>

```shell
$ kubectl get po red
NAME   READY   STATUS              RESTARTS   AGE
red    0/3     ContainerCreating   0          20s
$ kubectl get po blue -oyaml | grep name:
  name: blue
    name: teal
      name: kube-api-access-kwdhz
    name: navy
      name: kube-api-access-kwdhz
...
$ kubectl run yellow --image busybox --dry-run=client -oyaml > tmp.yaml
$ vim tmp.yaml
...
spec:
  containers:
  - image: busybox
    name: lemon
    command: ["sleep", "1000"]
  - image: redis
    name: gold
...
$ kubectl apply -f tmp.yaml
pod/yellow created
$ kubectl -n elastic-stack logs kibana | tail -n 1
{"type":"response","@timestamp":"2026-02-17T06:21:08Z","tags":[],"pid":1,"method":"get","statusCode":200,"req":{"url":"/plugins/security/images/logout.svg","method":"get","headers":{"host":"30601-port-7i5nxoqkdvfssxqn.labs.kodekloud.com","x-forwarded-proto":"http","x-forwarded-port":"443","x-forwarded-for":"122.32.255.46, 34.120.45.220, 169.254.169.126, 10.0.1.93","connection":"close","sec-ch-ua-platform":"\"macOS\"","user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36","sec-ch-ua":"\"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"144\", \"Google Chrome\";v=\"144\"","sec-ch-ua-mobile":"?0","accept":"image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8","sec-fetch-site":"same-origin","sec-fetch-mode":"no-cors","sec-fetch-dest":"image","referer":"https://30601-port-7i5nxoqkdvfssxqn.labs.kodekloud.com/app/kibana","accept-language":"ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7","priority":"i","x-cloud-trace-context":"e634831a0ea4987f2a699cf5f85fa210/5232735707314414303","via":"1.1 google","traceparent":"00-e634831a0ea4987f2a699cf5f85fa210-489e6966cdfa62df-00","forwarded":"for=\"122.32.255.46\";proto=https","accept-encoding":"gzip, deflate, br, zstd"},"remoteAddress":"172.17.0.1","userAgent":"172.17.0.1","referer":"https://30601-port-7i5nxoqkdvfssxqn.labs.kodekloud.com/app/kibana"},"res":{"statusCode":200,"responseTime":1,"contentLength":9},"message":"GET /plugins/security/images/logout.svg 200 1ms - 9.0B"}
$ kubectl get po app
NAME   READY   STATUS    RESTARTS   AGE
app    1/1     Running   0          9m43s
$ kubectl exec -it app -- cat /log/app.log | grep WARN | grep Login | tail -n 2
[2026-02-17 06:24:15,459] WARNING in event-simulator: USER5 Failed to Login as the account is locked due to MANY FAILED ATTEMPTS.
[2026-02-17 06:24:18,266] WARNING in event-simulator: USER5 Failed to Login as the account is locked due to MANY FAILED ATTEMPTS.
$ kubectl get po -n elastic-stack app -oyaml | sed '/^spec:/a\  initContainers:\n  - name: sidecar\n    image: kodekloud/filebeat-configured\n    restartPolicy: Always\n    volumeMounts:\n      - mountPath: /var/log/event-simulator\n        name: log-volume' | kubectl replace -n elastic-stack --force -f -
pod "app" deleted from elastic-stack namespace
pod/app replaced
```

</details>

<details>
<summary>
Readiness Probes
</summary>

```shell
$ ./curl-test.sh
Message from simple-webapp-1 : I am ready! OK
...
$ ./curl-test.sh
Message from simple-webapp-1 : I am ready! OK
Failed
...
$ kubectl get po simple-webapp-2 -oyaml | sed '/resources:.*/a\    readinessProbe:\n      httpGet:\n        path: \/ready\n        port: 8080' | sed '/^status:/,$ d' | kubectl replace --force -f -
pod "simple-webapp-2" deleted from default namespace
pod/simple-webapp-2 replaced
$ kubectl get po simple-webapp-1 -oyaml | sed '/resources:.*/a\    livenessProbe:\n      httpGet:\n        path: \/live\n        port: 8080\n      periodSeconds: 1\n      initialDelaySeconds: 80' | sed '/^status:/,$ d' | kubectl replace --force -f -
pod "simple-webapp-1" deleted from default namespace
pod/simple-webapp-1 replaced
$ kubectl get po simple-webapp-2 -oyaml | sed '/resources:.*/a\    livenessProbe:\n      httpGet:\n        path: \/live\n        port: 8080\n      periodSeconds: 1\n      initialDelaySeconds: 80' | sed '/^status:/,$ d' | kubectl replace --force -f -
```

</details>

<details>
<summary>
Logging
</summary>

```shell
$ kubectl logs webapp-1 | grep USER5
[2026-02-17 07:21:45,397] WARNING in event-simulator: USER5 Failed to Login as the account is locked due to MANY FAILED ATTEMPTS.
...
$ kubectl logs webapp-2 | grep WARN
Defaulted container "simple-webapp" out of: simple-webapp, db
...
[2026-02-17 07:23:53,586] WARNING in event-simulator: USER30 Order failed as the item is OUT OF STOCK.
```

</details>

<details>
<summary>
Monitoring
</summary>

```shell
$ kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
serviceaccount/metrics-server created
clusterrole.rbac.authorization.k8s.io/system:aggregated-metrics-reader created
clusterrole.rbac.authorization.k8s.io/system:metrics-server created
rolebinding.rbac.authorization.k8s.io/metrics-server-auth-reader created
clusterrolebinding.rbac.authorization.k8s.io/metrics-server:system:auth-delegator created
clusterrolebinding.rbac.authorization.k8s.io/system:metrics-server created
service/metrics-server created
deployment.apps/metrics-server created
apiservice.apiregistration.k8s.io/v1beta1.metrics.k8s.io created
$ kubectl top node
NAME           CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)
controlplane   186m         1%       885Mi           1%
node01         152m         0%       145Mi           0%
$ kubectl top po --sort-by=memory
NAME       CPU(cores)   MEMORY(bytes)
rabbit     100m         250Mi
elephant   13m          30Mi
lion       1m           16Mi
$ kubectl top po --sort-by=cpu
NAME       CPU(cores)   MEMORY(bytes)
rabbit     92m          250Mi
elephant   13m          30Mi
lion       1m           16Mi
```

</details>

<details>
<summary>
Init Containers
</summary>

```shell
$ kubectl get po -oyaml | grep initContainer -B30 | grep name
...
$ kubectl get po -oyaml | grep initContainer -A20 | grep image:
      image: busybox
      image: docker.io/library/busybox:latest
$ kubectl describe po blue | grep -i state: -A3
    State:          Terminated
      Reason:       Completed
      Exit Code:    0
      Started:      Tue, 17 Feb 2026 07:57:52 +0000
--
    State:          Running
      Started:      Tue, 17 Feb 2026 07:57:58 +0000
    Ready:          True
    Restart Count:  0
$ kubectl get po purple -oyaml
...
  initContainers:
  - command:
    - sh
    - -c
    - sleep 600
...
  - command:
    - sh
    - -c
    - sleep 1200
...
$ kubectl describe po purple | grep Status:
Status:           Pending
$ kubectl get po purple -oyaml | grep sleep
    - echo The app is running! && sleep 3600
    - sleep 600
    - sleep 1200
$ kubectl get po red -oyaml | sed '/^status:/,$d' | sed '/^spec/a\  initContainers:\n  - name: red-initcontainer\n    image: busybox\n    command: ["sleep", "20"]' | kubectl replace --force -f -
pod "red" deleted from default namespace
pod/red replaced
$ kubectl get po orange -oyaml | sed 's/sleeeep 2;/sleep 2/g' | kubectl replace --force -f -
pod "orange" deleted from default namespace
pod/orange replaced
```

</details>

## POD Design

<details>
<summary>
Labels and Selectors
</summary>

```shell
$ kubectl get po -l env=dev | wc -l
8
$ kubectl get po -l bu=finance | wc -l
7
$ kubectl get all -l env=prod
NAME              READY   STATUS    RESTARTS   AGE
pod/app-1-zzxdf   1/1     Running   0          117s
pod/app-2-lf7fc   1/1     Running   0          117s
pod/auth          1/1     Running   0          117s
pod/db-2-dvmtg    1/1     Running   0          117s

NAME            TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
service/app-1   ClusterIP   10.43.176.43   <none>        3306/TCP   117s

NAME                    DESIRED   CURRENT   READY   AGE
replicaset.apps/app-2   1         1         1       117s
replicaset.apps/db-2    1         1         1       117s
$ kubectl get po -l env=prod,bu=finance,tier=frontend
NAME          READY   STATUS    RESTARTS   AGE
app-1-zzxdf   1/1     Running   0          3m5s
$ cat replicaset-definition-1.yaml | sed 's/front-end/nginx/g' | kubectl apply -f -
replicaset.apps/replicaset-1 created
```

</details>

<details>
<summary>
Rolling Updates & Rollbacks
</summary>

```shell
$ ./curl-test.sh
Hello, Application Version: v1 ; Color: blue OK
...
$ kubectl get deploy
NAME       READY   UP-TO-DATE   AVAILABLE   AGE
frontend   4/4     4            4           70s
$ kubectl describe deploy | grep Image:
    Image:         kodekloud/webapp-color:v1
$ kubectl describe deploy | grep -i strategy
StrategyType:           RollingUpdate
RollingUpdateStrategy:  25% max unavailable, 25% max surge
$ kubectl set image deploy frontend simple-webapp=kodekloud/webapp-color:v2
deployment.apps/frontend image updated
$ kubectl get deploy frontend -oyaml | sed 's/type: RollingUpdate/type: Recreate/' | sed '/rollingUpdate:/,+2d' | kubectl apply -f -
deployment.apps/frontend configured
$ kubectl set image deploy frontend simple-webapp=kodekloud/webapp-color:v3
deployment.apps/frontend image updated
```

</details>

<details>
<summary>
Jobs and CronJobs
</summary>

```shell
$ kubectl apply -f throw-dice-pod.yaml
pod/throw-dice-pod created
$ kubectl create job throw-dice-job --image kodekloud/throw-dice
job.batch/throw-dice-job created
$ kubectl create job throw-dice-job --image kodekloud/throw-dice --dry-run=client -oyaml | sed '/^status:/,$d' | sed '$a\  backoffLimit: 6' | kubectl apply -f -
job.batch/throw-dice-job configured
$ kubectl describe job throw-dice-job | grep Status
Pods Statuses:    0 Active (0 Ready) / 1 Succeeded / 5 Failed
$ kubectl create job throw-dice-job --image kodekloud/throw-dice --dry-run=client -oyaml | sed '/^status:/,$d' | sed '$a\  backoffLimit: 36\n  completions: 2' | kubectl replace --force -f -
job.batch "throw-dice-job" deleted from default namespace
job.batch/throw-dice-job replaced
$ kubectl describe job throw-dice-job | grep Status
Pods Statuses:    0 Active (0 Ready) / 2 Succeeded / 2 Failed
$ kubectl create job throw-dice-job --image kodekloud/throw-dice --dry-run=client -oyaml | sed '/^status:/,$d' | sed '$a\  backoffLimit: 36\n  parallelism: 3\n  completions: 3' | kubectl replace --force -f -
job.batch "throw-dice-job" deleted from default namespace
job.batch/throw-dice-job replaced
$ kubectl create cj throw-dice-cron-job --image kodekloud/throw-dice --schedule "30 21 * * *"
cronjob.batch/throw-dice-cron-job created
```

</details>

## Services & Networking

<details>
<summary>
Kubernetes Services
</summary>

```shell
$ kubectl get svc
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.43.0.1    <none>        443/TCP   10m
$ kubectl describe svc kubernetes | grep -i target
TargetPort:               6443/TCP
$ kubectl describe svc kubernetes | grep -i label -A2
Labels:                   component=apiserver
                          provider=kubernetes
Annotations:              <none>
$ kubectl describe svc kubernetes | grep -i endpoint
Endpoints:                10.244.253.187:6443
$ kubectl get deploy | wc -l
2
$ kubectl describe deploy simple-webapp-deployment | grep Image:
    Image:         kodekloud/simple-webapp:red
$ cat service-definition-1.yaml | sed '5s/name:/& webapp-service/' | sed 's/type:/& NodePort/' | sed 's/targetPort:/& 8080/' | sed 's/port:/& 8080/' | sed 's/nodePort:/& 30080/' | sed 's/name: $/&simple-webapp/' | kubectl apply -f -
service/webapp-service created
```

</details>

<details>
<summary>
Network Policies
</summary>

```shell
$ kubectl get networkpolicy
NAME             POD-SELECTOR   AGE
payroll-policy   name=payroll   47s
$ kubectl get po -l name=payroll
NAME      READY   STATUS    RESTARTS   AGE
payroll   1/1     Running   0          84s
$ kubectl describe networkpolicy payroll-policy
Name:         payroll-policy
Namespace:    default
Created on:   2026-02-17 09:03:40 +0000 UTC
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
$ vim tmp.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: internal-policy
spec:
  podSelector:
    matchLabels:
      name: internal
  policyTypes:
  - Egress
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
```

</details>

<details>
<summary>
Ingress Networking - 1
</summary>

```shell
$ kubectl get po -A | grep ingress
ingress-nginx   ingress-nginx-admission-create-b54w6        0/1     Completed   0          2m
ingress-nginx   ingress-nginx-admission-patch-bqcc8         0/1     Completed   0          2m
ingress-nginx   ingress-nginx-controller-788776ff65-t72pw   1/1     Running     0          2m
$ kubectl get ing -A
NAMESPACE   NAME                 CLASS    HOSTS   ADDRESS          PORTS   AGE
app-space   ingress-wear-watch   <none>   *       172.20.170.203   80      73s
$ kubectl get deploy -n app-space
NAME              READY   UP-TO-DATE   AVAILABLE   AGE
default-backend   1/1     1            1           3m
webapp-video      1/1     1            1           3m
webapp-wear       1/1     1            1           3m
$ kubectl describe ing -n app-space ingress-wear-watch
Name:             ingress-wear-watch
Labels:           <none>
Namespace:        app-space
Address:          172.20.170.203
Ingress Class:    <none>
Default backend:  <default>
Rules:
  Host        Path  Backends
  ----        ----  --------
  *
              /wear    wear-service:8080 (172.17.0.4:8080)
              /watch   video-service:8080 (172.17.0.5:8080)
Annotations:  nginx.ingress.kubernetes.io/rewrite-target: /
              nginx.ingress.kubernetes.io/ssl-redirect: false
Events:
  Type    Reason  Age                    From                      Message
  ----    ------  ----                   ----                      -------
  Normal  Sync    7m10s (x2 over 7m11s)  nginx-ingress-controller  Scheduled for sync
$ kubectl get deploy -n ingress-nginx ingress-nginx-controller -oyaml | grep -i default
        - --default-backend-service=app-space/default-backend-service
      schedulerName: default-scheduler
          defaultMode: 420
$ kubectl edit ing -n app-space ingress-wear-watch
      - backend:
          service:
            name: video-service
            port:
              number: 8080
        path: /stream
        pathType: Prefix
...
ingress.networking.k8s.io/ingress-wear-watch edited
$ kubectl edit ing -n app-space ingress-wear-watch
      - backend:
          service:
            name: food-service
            port:
              number: 8080
        path: /eat
        pathType: Prefix
...
ingress.networking.k8s.io/ingress-wear-watch edited
$ kubectl get svc -A | grep pay
critical-space   pay-service                          ClusterIP   172.20.193.117   <none>        8282/TCP                     19s
$ kubectl get deploy -n critical-space
NAME         READY   UP-TO-DATE   AVAILABLE   AGE
webapp-pay   1/1     1            1           94s
$ kubectl get ing -n app-space ingress-wear-watch -oyaml > tmp.yaml
$ vim tmp.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
  name: critical-ingress
  namespace: critical-space
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
$ kubectl apply -f tmp.yaml
ingress.networking.k8s.io/critical-ingress created
```

</details>

<details>
<summary>
Ingress Networking - 2
</summary>

```shell
$ vim tmp.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-app-ingress
  namespace: webapp
spec:
  ingressClassName: nginx
  rules:
  - host: app.kodekloud.local
    http:
      paths:
      - backend:
          service:
            name: web-app
            port:
              number: 80
        path: /
        pathType: Prefix
$ kubectl apply -f tmp.yaml
ingress.networking.k8s.io/web-app-ingress created
$ vim tmp.yaml
...
  tls:
  - hosts:
      - app.kodekloud.local
    secretName: app-tls
...
$ kubectl apply -f tmp.yaml
ingress.networking.k8s.io/web-app-ingress configured
$ vim tmp.yaml
...
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
...
$ kubectl apply -f tmp.yaml
ingress.networking.k8s.io/web-app-ingress configured
```

</details>

## State Persistence

<details>
<summary>
Persistent Volumes
</summary>

```shell
$ kubectl get po webapp -oyaml | sed '/^status:/,$d' > tmp.yaml
$ vim tmp.yaml
...
    - mountPath: /log
      name: log-volume
...
  - name: log-volume
    hostPath:
      path: /var/log/webapp
      type: Directory
...
$ kubectl replace --force -f tmp.yaml
pod "webapp" deleted from default namespace
pod/webapp replaced
$ vim tmp.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-log
spec:
  capacity:
    storage: 100Mi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "/pv/log"
  persistentVolumeReclaimPolicy: Retain
$ kubectl apply -f tmp.yaml
persistentvolume/pv-log created
$ vim tmp.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: claim-log-1
spec:
  resources:
    requests:
      storage: 50Mi
  accessModes:
  - ReadWriteOnce
$ kubectl apply -f tmp.yaml
persistentvolumeclaim/claim-log-1 created
$ kubectl get pvc
NAME          STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
claim-log-1   Pending                                                     <unset>                 19s
$ kubectl get pv
NAME     CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      CLAIM   STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-log   100Mi      RWX            Retain           Available                          <unset>                          3m40s
$ vim tmp.yaml
...
  accessModes:
  - ReadWriteMany
$ kubectl replace --force -f tmp.yaml
persistentvolumeclaim "claim-log-1" deleted from default namespace
persistentvolumeclaim/claim-log-1 replaced
$ kubectl get po webapp -oyaml | sed '/^status:/,$d' | sed 's/- hostPath:/- persistentVolumeClaim:\n      claimName: claim-log-1/' | sed '\|path: /var/log/webapp|,+1d' | kubectl replace --force -f -
pod "webapp" deleted from default namespace
pod/webapp replaced
$ kubectl describe pv pv-log | grep -i policy
Reclaim Policy:  Retain
$ kubectl delete pvc claim-log-1
persistentvolumeclaim "claim-log-1" deleted from default namespace
^C
$ kubectl delete po webapp
pod "webapp" deleted from default namespace
$ kubectl get pvc
No resources found in default namespace.
$ kubectl get pv
NAME     CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS     CLAIM                 STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-log   100Mi      RWX            Retain           Released   default/claim-log-1                  <unset>                          21m
```

</details>

<details>
<summary>
Storage Class
</summary>

```shell
$ kubectl get sc | wc -l
2
$ kubectl get sc | wc -l
4
$ kubectl describe sc | grep -i provisioner:
Provisioner:           rancher.io/local-path
Provisioner:           kubernetes.io/no-provisioner
Provisioner:           kubernetes.io/portworx-volume
$ kubectl get sc
NAME                        PROVISIONER                     RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
local-path (default)        rancher.io/local-path           Delete          WaitForFirstConsumer   false                  9m41s
local-storage               kubernetes.io/no-provisioner    Delete          WaitForFirstConsumer   false                  2m50s
portworx-io-priority-high   kubernetes.io/portworx-volume   Delete          Immediate              false                  2m50s
$ vim tmp.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: local-pvc
spec:
  storageClassName: local-path
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Mi
$ kubectl apply -f tmp.yaml
persistentvolumeclaim/local-pvc created
$ kubectl get pvc
NAME        STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
local-pvc   Pending                                      local-path     <unset>                 19s
$ kubectl describe pvc local-pvc | tail -n 1
  Normal  WaitForFirstConsumer  7s (x5 over 67s)  persistentvolume-controller  waiting for first consumer to be created before binding
$ kubectl run nginx --image nginx:alpine --dry-run=client -oyaml > tmp.yaml
$ vim tmp.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: nginx
  name: nginx
spec:
  containers:
  - image: nginx:alpine
    name: nginx
    resources: {}
    volumeMounts:
    - mountPath: "/var/www/html"
      name: local-pvc
  volumes:
  - name: local-pvc
    persistentVolumeClaim:
      claimName: local-pvc
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
$ kubectl apply -f tmp.yaml
pod/nginx created
$ kubectl get pvc
NAME        STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
local-pvc   Bound    pvc-440842b5-d1a8-4c6d-b351-16a1d0676f06   500Mi      RWO            local-path     <unset>                 4m2s
$ kubectl get sc local-storage -oyaml | sed '/annotations:/,+3d' | sed 's/name:.*/name: delayed-volume-sc/' | sed '/resourceVersion:/,+1d' | kubectl apply -f -
storageclass.storage.k8s.io/delayed-volume-sc created
```

</details>

## Updates for Sep 2021 Changes

<details>
<summary>
Practice test Docker Images
</summary>

```shell
$ docker images | wc -l
10
$ docker images | grep ubuntu
ubuntu                          latest    97bed23a3497   4 months ago   78.1MB
$ docker images | grep nginx
nginx                           latest        657fdcd1c365   4 months ago   152MB
nginx                           alpine        5e7abcdd2021   4 months ago   52.7MB
nginx                           1.14-alpine   8a2fb25a19f5   6 years ago    16MB
$ cat webapp-color/Dockerfile | head -n 1
FROM python:3.6
$ cat webapp-color/Dockerfile | grep COPY
COPY . /opt/
$ cat webapp-color/Dockerfile | grep RUN
RUN pip install flask
$ cat webapp-color/Dockerfile | tail -n 1
ENTRYPOINT ["python", "app.py"]
$ cat webapp-color/Dockerfile | grep EXPOSE
EXPOSE 8080
$ cd webapp-color && docker build -t webapp-color .
...
Step 6/6 : ENTRYPOINT ["python", "app.py"]
 ---> Running in 773a023b637e
 ---> Removed intermediate container 773a023b637e
 ---> d844a7a5f40a
Successfully built d844a7a5f40a
Successfully tagged webapp-color:latest
$ docker run -p 8282:8080 webapp-color
 This is a sample web application that displays a colored background.
 A color can be specified in two ways.
...
$ docker run python:3.6 cat /etc/os-release | grep NAME
PRETTY_NAME="Debian GNU/Linux 11 (bullseye)"
NAME="Debian GNU/Linux"
VERSION_CODENAME=bullseye
$ docker images | grep webapp-color
webapp-color                    latest        d844a7a5f40a   4 minutes ago   913MB
$ cat Dockerfile | sed 's/3.6/3.6-slim/' > Dockerfile.slim
$ docker build -f Dockerfile.slim -t webapp-color:lite .
Step 6/6 : ENTRYPOINT ["python", "app.py"]
 ---> Running in 50ced252f965
 ---> Removed intermediate container 50ced252f965
 ---> e191a3338d2d
Successfully built e191a3338d2d
Successfully tagged webapp-color:lite
$ docker images | grep webapp-color
webapp-color                    lite          e191a3338d2d   38 seconds ago   130MB
webapp-color                    latest        d844a7a5f40a   8 minutes ago    913MB
$ docker run -p 8383:8080 webapp-color:lite
 This is a sample web application that displays a colored background.
 A color can be specified in two ways.
...
```

</details>

<details>
<summary>
Practice Test KubeConfig
</summary>

```shell
$ kubectl config get-clusters
NAME
kubernetes
$ kubectl config get-users
NAME
kubernetes-admin
$ kubectl config get-contexts
CURRENT   NAME                          CLUSTER      AUTHINFO           NAMESPACE
*         kubernetes-admin@kubernetes   kubernetes   kubernetes-admin
$ kubectl config get-clusters --kubeconfig my-kube-config
NAME
production
development
kubernetes-on-aws
test-cluster-1
$ kubectl config get-contexts --kubeconfig my-kube-config
CURRENT   NAME                         CLUSTER             AUTHINFO    NAMESPACE
          aws-user@kubernetes-on-aws   kubernetes-on-aws   aws-user
          research                     test-cluster-1      dev-user
*         test-user@development        development         test-user
          test-user@production         production          test-user
$ kubectl config view --kubeconfig my-kube-config  | grep aws-user
    user: aws-user
  name: aws-user@kubernetes-on-aws
- name: aws-user
    client-certificate: /etc/kubernetes/pki/users/aws-user/aws-user.crt
    client-key: /etc/kubernetes/pki/users/aws-user/aws-user.key
$ kubectl config use-context --kubeconfig my-kube-config research
Switched to context "research".
$ echo "export KUBECONFIG=~/my-kube-config" >> ~/.bashrc
$ source ~/.bashrc
$ kubectl get node
error: unable to read client-cert /etc/kubernetes/pki/users/dev-user/developer-user.crt for dev-user due to open /etc/kubernetes/pki/users/dev-user/developer-user.crt: no such file or directory
$ sed -i 's/developer-user.crt/dev-user.crt/g' my-kube-config
$ kubectl get node
NAME           STATUS   ROLES           AGE   VERSION
controlplane   Ready    control-plane   21m   v1.34.0
```

</details>

<details>
<summary>
Practice Test Role Based Access Controls
</summary>

```shell
$ kubectl get -n kube-system po kube-apiserver-controlplane -oyaml | grep -i auth
    - --authorization-mode=Node,RBAC
    - --enable-bootstrap-token-auth=true
$ kubectl get role
No resources found in default namespace.
$ kubectl get role -A | wc -l
13
$ kubectl describe role -n kube-system kube-proxy
Name:         kube-proxy
Labels:       <none>
Annotations:  <none>
PolicyRule:
  Resources   Non-Resource URLs  Resource Names  Verbs
  ---------   -----------------  --------------  -----
  configmaps  []                 [kube-proxy]    [get]
$ kubectl describe rolebinding -n kube-system kube-proxy
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
$ kubectl create role developer --resource po --verb list,create,delete
role.rbac.authorization.k8s.io/developer created
$ kubectl create rolebinding dev-user-binding --role developer --user dev-user
rolebinding.rbac.authorization.k8s.io/dev-user-binding created
$ kubectl get role -n blue developer -oyaml | sed '/blue-app/a\  - dark-blue-app' | kubectl apply -f -
role.rbac.authorization.k8s.io/developer configured
$ kubectl get role -n blue developer -oyaml | sed '$a\- apiGroups:\n  - apps\n  resources:\n  - deployments\n  verbs:\n  - create' | kubectl apply -f -
role.rbac.authorization.k8s.io/developer configured
```

</details>

<details>
<summary>
Practice Test Cluster Roles
</summary>

```shell
$ kubectl get clusterrole | wc -l
77
$ kubectl get clusterrolebinding | wc -l
62
$ kubectl describe clusterrolebinding cluster-admin
Name:         cluster-admin
Labels:       kubernetes.io/bootstrapping=rbac-defaults
Annotations:  rbac.authorization.kubernetes.io/autoupdate: true
Role:
  Kind:  ClusterRole
  Name:  cluster-admin
Subjects:
  Kind   Name            Namespace
  ----   ----            ---------
  Group  system:masters
$ kubectl describe clusterrole cluster-admin
Name:         cluster-admin
Labels:       kubernetes.io/bootstrapping=rbac-defaults
Annotations:  rbac.authorization.kubernetes.io/autoupdate: true
PolicyRule:
  Resources  Non-Resource URLs  Resource Names  Verbs
  ---------  -----------------  --------------  -----
  *.*        []                 []              [*]
             [*]                []              [*]
$ kubectl create clusterrole michelle --resource node --verb get,list
clusterrole.rbac.authorization.k8s.io/michelle created
$ kubectl create clusterrolebinding michelle --clusterrole michelle --user michelle
clusterrolebinding.rbac.authorization.k8s.io/michelle created
$ kubectl auth can-i list node --as michelle
yes
$ kubectl create clusterrole storage-admin --resource pv,sc --verb get,list
clusterrole.rbac.authorization.k8s.io/storage-admin created
$ kubectl create clusterrolebinding michelle-storage-admin --clusterrole storage-admin --user michelle
clusterrolebinding.rbac.authorization.k8s.io/michelle-storage-admin created
```

</details>

<details>
<summary>
Labs - Admission Controllers
</summary>

```shell
$ sed -i 's/true/false/' /etc/kubernetes/imgvalidation/imagepolicy-conf.yaml
$ sed -i 's|server:.*|server: https://image-checker-webhook.default.svc:1323/image_policy|' /etc/kubernetes/imgvalidation/kubeconf.yaml
$ sed -i 's/enable-admission-plugins=NodeRestriction/&,ImagePolicyWebhook/' /etc/kubernetes/manifests/kube-apiserver.yaml
$ sed -i '/- kube-apiserver/a\    - --admission-control-config-file=/etc/kubernetes/imgvalidation/admission-configuration.yaml' /etc/kubernetes/manifests/kube-apiserver.yaml
$ vim /etc/kubernetes/manifests/kube-apiserver.yaml
...
    - name: imgvalidation
      mountPath: /etc/kubernetes/imgvalidation
      readOnly: true
...
  - name: imgvalidation
    hostPath:
      path: /etc/kubernetes/imgvalidation
      type: Directory
...
$ kubectl apply -f ~/test-deploy.yaml
Error from server (Forbidden): error when creating "/root/test-deploy.yaml": pods "test-deploy" is forbidden: Post "https://image-checker-webhook.default.svc:1323/image_policy?timeout=30s": dial tcp: lookup image-checker-webhook.default.svc on 10.96.0.10:53: no such host
```

</details>

<details>
<summary>
Labs - Validating and Mutating Admission Controllers
</summary>

```shell
$ kubectl create ns webhook-demo
namespace/webhook-demo created
$ kubectl -n webhook-demo create secret tls webhook-server-tls --cert "/root/keys/webhook-server-tls.crt" --key "/root/keys/webhook-server-tls.key"
secret/webhook-server-tls created
$ kubectl create -f webhook-deployment.yaml
deployment.apps/webhook-server created
$ kubectl create -f webhook-service.yaml
service/webhook-server created
$ kubectl apply -f webhook-configuration.yaml
mutatingwebhookconfiguration.admissionregistration.k8s.io/demo-webhook created
$ kubectl apply -f pod-with-defaults.yaml
pod/pod-with-defaults created
$ kubectl get pod pod-with-defaults -oyaml | grep security -A2
  securityContext:
    runAsNonRoot: true
    runAsUser: 1234
$ kubectl apply -f pod-with-override.yaml
pod/pod-with-override created
$ kubectl apply -f pod-with-conflict.yaml
Error from server: error when creating "pod-with-conflict.yaml": admission webhook "webhook-server.webhook-demo.svc" denied the request: runAsNonRoot specified, but runAsUser set to 0 (the root user)
```

</details>

<details>
<summary>
Lab - API Versions/Deprecations
</summary>

```shell
$ kubectl api-resources | grep authorization
localsubjectaccessreviews                        authorization.k8s.io/v1           true         LocalSubjectAccessReview
selfsubjectaccessreviews                         authorization.k8s.io/v1           false        SelfSubjectAccessReview
selfsubjectrulesreviews                          authorization.k8s.io/v1           false        SelfSubjectRulesReview
subjectaccessreviews                             authorization.k8s.io/v1           false        SubjectAccessReview
clusterrolebindings                              rbac.authorization.k8s.io/v1      false        ClusterRoleBinding
clusterroles                                     rbac.authorization.k8s.io/v1      false        ClusterRole
rolebindings                                     rbac.authorization.k8s.io/v1      true         RoleBinding
roles                                            rbac.authorization.k8s.io/v1      true         Role
$ cp -v /etc/kubernetes/manifests/kube-apiserver.yaml /root/kube-apiserver.yaml.backup
'/etc/kubernetes/manifests/kube-apiserver.yaml' -> '/root/kube-apiserver.yaml.backup'
$ sed -i '\|- kube-apiserver|a\    - --runtime-config=rbac.authorization.k8s.io/v1alpha1' /etc/kubernetes/manifests/kube-apiserver.yaml
$ curl -LO https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl-convert
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 55.8M  100 55.8M    0     0  65.2M      0 --:--:-- --:--:-- --:--:-- 65.2M
$ chmod +x kubectl-convert
$ mv kubectl-convert /usr/local/bin/kubectl-convert
$ kubectl-convert -h
Convert config files between different API versions. Both YAML and JSON formats are accepted.

 The command takes filename, directory, or URL as input, and convert it into format of version specified by --output-version flag. If target version is not specified or not supported, convert to latest version.
$ kubectl convert -f ingress-old.yaml --output-version networking.k8s.io/v1 | kubectl apply -f -
ingress.networking.k8s.io/ingress-space created
```

</details>

<details>
<summary>
Practice Test - Custom Resource Definition
</summary>

```shell
$ cat crd.yaml | sed '5s/name:/& internals.datasets.kodekloud.com/' | sed 's/group:/& datasets.kodekloud.com/' | sed 's/scope:/& Namespaced/' | sed 's/name: v2/name: v1/' | sed 's/served: false/served: true/' | sed 's/plural: internal/&s/' | kubectl apply -f -
customresourcedefinition.apiextensions.k8s.io/internals.datasets.kodekloud.com created
$ kubectl apply -f custom.yaml
internal.datasets.kodekloud.com/internal-space created
$ kubectl describe crd collectors | grep -i properties -A 6
        Properties:
          Spec:
            Properties:
              Image:
                Type:  string
              Name:
                Type:  string
              Replicas:
                Type:  integer
$ vim tmp.yaml
kind: Global
apiVersion: traffic.controller/v1
metadata:
  name: datacenter
spec:
  dataField: 2
  access: true
$ kubectl apply -f tmp.yaml
global.traffic.controller/datacenter created
$ kubectl api-resources | grep global
globals                             gb           traffic.controller/v1             true         Global
```

</details>

<details>
<summary>
Practice Test - Deployment strategies
</summary>

```shell
$ kubectl describe deploy frontend | grep -i strategy
StrategyType:           RollingUpdate
RollingUpdateStrategy:  25% max unavailable, 25% max surge
$ kubectl get svc -owide
NAME               TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE     SELECTOR
frontend           NodePort    172.20.136.85   <none>        8080:30081/TCP   65s     app=frontend-application
frontend-service   NodePort    172.20.85.4     <none>        8080:30080/TCP   65s     app=frontend
kubernetes         ClusterIP   172.20.0.1      <none>        443/TCP          5m10s   <none>
$ kubectl get po -l app=frontend
NAME                        READY   STATUS    RESTARTS   AGE
frontend-745f74d569-5wfgb   1/1     Running   0          93s
frontend-745f74d569-8qdsn   1/1     Running   0          93s
frontend-745f74d569-czn6h   1/1     Running   0          93s
frontend-745f74d569-g2tb7   1/1     Running   0          93s
frontend-745f74d569-rf2nz   1/1     Running   0          93s
$ kubectl scale deploy frontend-v2 --replicas 1
deployment.apps/frontend-v2 scaled
$ kubectl scale deploy frontend --replicas 0
deployment.apps/frontend scaled
$ kubectl scale deploy frontend-v2 --replicas 5
deployment.apps/frontend-v2 scaled
$ kubectl delete deploy frontend
deployment.apps "frontend" deleted from default namespace
```

</details>

<details>
<summary>
Labs - Install Helm
</summary>

```shell
$ cat /etc/os-release | grep NAME
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_CODENAME=jammy
UBUNTU_CODENAME=jammy
$ sudo apt-get install curl gpg apt-transport-https -y
...
$ curl -fsSL https://packages.buildkite.com/helm-linux/helm-debian/gpgkey | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
$ echo "deb [signed-by=/usr/share/keyrings/helm.gpg] https://packages.buildkite.com/helm-linux/helm-debian/any/ any main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
$ sudo apt-get update
...
Get:7 https://packages.buildkite.com/helm-linux/helm-debian/any any InRelease [29.3 kB]
Get:8 https://packages.buildkite.com/helm-linux/helm-debian/any any/main amd64 Packages [2,513 B]
...
$ sudo apt-get install helm -y
...
$ helm -h | grep -i env
Environment variables:
- If a HELM_*_HOME environment variable is set, it will be used
  env         helm client environment information
$ helm version
version.BuildInfo{Version:"v3.19.2", GitCommit:"8766e718a0119851f10ddbe4577593a45fadf544", GitTreeState:"clean", GoVersion:"go1.24.9"}
$ helm -h | grep verbose
      --debug                           enable verbose output
```

</details>

<details>
<summary>
Labs - Helm Concepts
</summary>

```shell
$ helm repo add bitnami https://charts.bitnami.com/bitnami
"bitnami" has been added to your repositories
$ helm search repo joomla
NAME            CHART VERSION   APP VERSION     DESCRIPTION
bitnami/joomla  20.0.4          5.1.2           DEPRECATED Joomla! is an award winning open sou...
$ helm repo list
NAME            URL
bitnami         https://charts.bitnami.com/bitnami
puppet          https://puppetlabs.github.io/puppetserver-helm-chart
hashicorp       https://helm.releases.hashicorp.com
$ helm install bravo bitnami/drupal
Pulled: us-central1-docker.pkg.dev/kk-lab-prod/helm-charts/bitnami/drupal:21.1.3
Digest: sha256:1986543cf00e9b7ec2d03c97e5d11588efd6419767036b3555afd8b9c3203f79
I0217 16:23:24.556621   13157 warnings.go:110] "Warning: spec.SessionAffinity is ignored for headless services"
NAME: bravo
LAST DEPLOYED: Tue Feb 17 16:23:24 2026
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
CHART NAME: drupal
CHART VERSION: 21.1.3
APP VERSION: 11.1.2
...
$ helm uninstall bravo
release "bravo" uninstalled
$ helm search repo apache
NAME                            CHART VERSION   APP VERSION     DESCRIPTION
bitnami/apache                  11.4.29         2.4.65          Apache HTTP Server is an open-source HTTP serve...
...
$ helm pull --untar bitnami/apache --version 10.1.1
Pulled: us-central1-docker.pkg.dev/kk-lab-prod/helm-charts/bitnami/apache:10.1.1
Digest: sha256:6a126989a295a2a1b28348d7320b828138c355bc739017b7be70a982c924e8a5
$ helm install mywebapp bitnami/apache --set replicaCount=2,service.type=NodePort,service.nodePorts.http=30080
Pulled: us-central1-docker.pkg.dev/kk-lab-prod/helm-charts/bitnami/apache:11.3.2
Digest: sha256:1bd45c97bb7a0000534e3abc5797143661e34ea7165aa33068853c567e6df9f2
NAME: mywebapp
LAST DEPLOYED: Tue Feb 17 16:35:06 2026
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
CHART NAME: apache
CHART VERSION: 11.3.2
APP VERSION: 2.4.63
...
```

</details>

<details>
<summary>
Lab - Managing Directories 🆕
</summary>

```shell
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - db/db-config.yaml
  - db/db-depl.yaml
  - db/db-service.yaml
  - message-broker/rabbitmq-config.yaml
  - message-broker/rabbitmq-depl.yaml
  - message-broker/rabbitmq-service.yaml
  - nginx/nginx-depl.yaml
  - nginx/nginx-service.yaml
$ kubectl apply -k code/k8s
configmap/db-credentials created
configmap/redis-credentials created
service/db-service created
service/nginx-service created
service/rabbit-cluster-ip-service created
deployment.apps/db-deployment created
deployment.apps/nginx-deployment created
deployment.apps/rabbitmq-deployment created
$ kubectl get po | wc -l
6
$ kubectl get svc rabbit-cluster-ip-service -owide
NAME                        TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE     SELECTOR
rabbit-cluster-ip-servic
$ vim code/k8s/db/kustomization.yaml
resources:
  - db-depl.yaml
  - db-service.yaml
  - db-config.yaml
$ vim code/k8s/message-broker/kustomization.yaml
resources:
  - rabbitmq-config.yaml
  - rabbitmq-depl.yaml
  - rabbitmq-service.yaml
$ vim code/k8s/nginx/kustomization.yaml
resources:
  - nginx-depl.yaml
  - nginx-service.yaml
$ vim code/k8s/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - db/
  - message-broker/
  - nginx/
$ kubectl apply -k /root/code/k8s
configmap/db-credentials created
configmap/redis-credentials created
service/db-service created
service/nginx-service created
service/rabbit-cluster-ip-service created
deployment.apps/db-deployment created
deployment.apps/nginx-deployment created
deployment.apps/rabbitmq-deployment created
$ kubectl get po | wc -l
7
```

</details>

<details>
<summary>
Lab - Transformers 🆕
</summary>

```shell
$ cat code/k8s/kustomization.yaml | grep -i label -A1
commonLabels:
  sandbox: dev
$ cat code/k8s/db/kustomization.yaml | grep -i prefix
namePrefix: data-
$ cat code/k8s/monitoring/kustomization.yaml | grep namespace
namespace: logging
$ sed -i '$a\\ncommonAnnotations:\n  owner: bob@gmail.com' code/k8s/nginx/kustomization.yaml
$ sed -i '$a\\ncommonAnnotations:\n  owner: bob@gmail.com' code/k8s/monitoring/kustomization.yaml
$ sed -i '$a\\nimages:\n  - name: postgres\n    newName: mysql' code/k8s/kustomization.yaml
$ sed -i '$a\\nimages:\n  - name: nginx\n    newTag: "1.23"' code/k8s/nginx/kustomization.yaml
```

</details>

<details>
<summary>
Lab - Patches 🆕
</summary>

```shell
$ cat code/k8s/kustomization.yaml | grep nginx-deploy -A4
      name: nginx-deployment
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 3
$ cat code/k8s/mongo-depl.yaml | grep labels -A1
      labels:
        component: mongo
$ cat code/k8s/mongo-label-patch.yaml
- op: add
  path: /spec/template/metadata/labels/cluster
  value: staging

- op: add
  path: /spec/template/metadata/labels/feature
  value: db
$ cat code/k8s/kustomization.yaml | grep -i targetport -A1
        path: /spec/ports/0/targetPort
        value: 30000
$ cat code/k8s/api-* | grep containers -A 2
      containers:
        - name: nginx
          image: nginx
--
      containers:
        - name: memcached
          image: memcached
$ cat code/k8s/mongo-* | grep -i volume -A1
          volumeMounts:
            - mountPath: /data/db
              name: mongo-volume
      volumes:
        - name: mongo-volume
          persistentVolumeClaim:
            claimName: host-pvc
$ vim code/k8s/api-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-deployment
spec:
  template:
    spec:
      containers:
      - $patch: delete
        name: memcached
$ kubectl apply -k code/k8s
service/mongo-cluster-ip-service created
deployment.apps/api-deployment created
deployment.apps/mongo-deployment created
$ vim code/k8s/kustomization.yaml
...
patches:
  - target:
      kind: Deployment
      name: mongo-deployment
    patch: |-
      - op: remove
        path: /spec/template/metadata/labels/org
$ kubectl apply -k code/k8s
service/mongo-cluster-ip-service created
deployment.apps/api-deployment created
deployment.apps/mongo-deployment created
```

</details>

<details>
<summary>
Lab - Overlay 🆕
</summary>

```shell
$ kubectl kustomize code/k8s/overlays/prod | grep api-deploy -A20 | grep image
        image: memcached
$ kubectl kustomize code/k8s/overlays/prod | grep api-deploy -A20 | grep replicas
  replicas: 2
$ kubectl kustomize code/k8s/overlays/staging | grep MONGO_INITDB_ROOT_PASSWORD -A4
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            configMapKeyRef:
              key: password
              name: db-creds
$ kubectl kustomize code/k8s/overlays/staging | grep password
  password: superp@ssword123
              key: password
$ kubectl kustomize code/k8s/overlays/prod | grep replicas
  replicas: 2
  replicas: 1
  replicas: 2
$ kubectl kustomize code/k8s/overlays/dev | grep api-deploy -A50 | grep nginx -B50
  name: api-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      component: api
  template:
    metadata:
      labels:
        component: api
    spec:
      containers:
      - env:
        - name: DB_USERNAME
          valueFrom:
            configMapKeyRef:
              key: username
              name: db-creds
        - name: DB_PASSWORD
          valueFrom:
            configMapKeyRef:
              key: password
              name: db-creds
        - name: DB_CONNECTION
          value: db.kodekloud.com
        image: nginx
$ vim code/k8s/overlays/QA/kustomization.yaml
patches:
  - target:
      kind: Deployment
      name: api-deployment
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/image
        value: caddy
$ kubectl apply -k code/k8s/overlays/QA
configmap/db-creds created
deployment.apps/api-deployment created
deployment.apps/mongo-deployment created
$ kubectl create deploy mysql-deployment --image mysql --dry-run=client -oyaml > code/k8s/overlays/staging/mysql-depl.yaml
$ sed -i '/image: mysql/a\        env:\n        - name: MYSQL_ROOT_PASSWORD\n          value: mypassword' code/k8s/overlays/staging/mysql-depl.yaml
$ sed -i '$a\\nresources:\n  - mysql-depl.yaml' code/k8s/overlays/staging/kustomization.yaml
$ kubectl apply -k code/k8s/overlays/staging
deployment.apps/mysql-deployment created
```

</details>

<details>
<summary>
Lab - Components 🆕
</summary>

```shell
$ cat code/project_mercury/overlays/community/kustomization.yaml
bases:
  - ../../base

components:
  - ../../components/auth
$ cat code/project_mercury/overlays/devkkustomization.yaml
bases:
  - ../../base

components:
  - ../../components/auth
  - ../../components/db
  - ../../components/logging
$ cat code/project_mercury/components/db/api-patch.yaml | grep env -A20
          env:
            - name: DB_CONNECTION
              value: postgres-service
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-creds
                  key: password
$ cat code/project_mercury/components/db/kustomization.yaml | grep secret -A3
secretGenerator:
  - name: db-creds
    literals:
      - password=password1
$ sed -i '$a\  - ../../components/logging' code/project_mercury/overlays/community/kustomization.yaml
$ kubectl apply -k code/project_mercury/overlays/community
service/api-service created
service/keycloak-service created
service/prometheus-service created
deployment.apps/api-deployment created
deployment.apps/keycloak-deployment created
deployment.apps/prometheus-deployment created
$ vim code/project_mercury/components/caching/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1alpha1
kind: Component
resources:
  - redis-depl.yaml
  - redis-service.yaml
$ cat code/project_mercury/base/api-depl.yaml | sed '/replicas:/,+3d' | sed '7,9d' | sed 's/image:.*/env:\n          - name: REDIS_CONNECTION\n            value: redis-service/' > code/project_mercury/components/caching/api-patch.yaml
$ sed -i '$a\patches:\n  - path: api-patch.yaml' code/project_mercury/components/caching/kustomization.yaml
$ sed -i '/resources:/a\  - ../../base' code/project_mercury/components/caching/kustomization.yaml
$ kubectl apply -k code/project_mercury/components/caching
service/api-service created
service/redis-service created
deployment.apps/api-deployment created
deployment.apps/redis-deployment created
$ sed -i '$a\  - ../../components/caching' code/project_mercury/overlays/enterprise/kustomization.yaml
$ sed -i '\|../../base|d' code/project_mercury/components/caching/kustomization.yaml
$ kubectl apply -k code/project_mercury/overlays/enterprise
secret/db-creds-dd6525th4g created
service/api-service created
service/keycloak-service created
service/postgres-service created
service/redis-service created
deployment.apps/api-deployment created
deployment.apps/keycloak-deployment created
deployment.apps/postgres-deployment created
deployment.apps/redis-deployment created
```

</details>

## Lightning Labs

<details>
<summary>
Lab: Lightning Lab - 1
</summary>

```shell
$ kubectl run logger --image nginx:alpine --dry-run=client -oyaml >> 1.yaml
$ vim 1.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: log-volume
spec:
  storageClassName: manual
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "/opt/volume/nginx"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: log-claim
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 200Mi
---
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: logger
  name: logger
spec:
  containers:
  - image: nginx:alpine
    name: logger
    resources: {}
    volumeMounts:
      - name: log
        mountPath: "/var/www/nginx"
  volumes:
    - name: log
      persistentVolumeClaim:
        claimName: log-claim
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
$ kubectl apply -f 1.yaml
persistentvolume/log-volume created
persistentvolumeclaim/log-claim created
pod/logger created
$ kubectl get po logger
NAME     READY   STATUS    RESTARTS   AGE
logger   1/1     Running   0          33s

$ kubectl get po --show-labels
NAME           READY   STATUS    RESTARTS   AGE    LABELS
logger         1/1     Running   0          88s    run=logger
secure-pod     1/1     Running   0          71s    run=secure-pod
webapp-color   1/1     Running   0          2m3s   name=webapp-color
$ vim 2.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: test-network-policy
  namespace: default
spec:
  podSelector:
    matchLabels:
      run: secure-pod
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          name: webapp-color
    ports:
    - protocol: TCP
      port: 80
$ kubectl apply -f 2.yaml
networkpolicy.networking.k8s.io/test-network-policy created
$ kubectl exec -it webapp-color -- nc -vz secure-service 80
secure-service (172.20.6.40:80) open

$ kubectl create ns dvl1987
namespace/dvl1987 created
$ kubectl create cm -n dvl1987 time-config --from-literal TIME_FREQ=10
configmap/time-config created
$ kubectl run time-check -n dvl1987 --image busybox --command --dry-run=client -oyaml -- /bin/sh -c 'while true; do date; sleep $TIME_FREQ; done > /opt/time/time-check.log' > 3.yaml
$ vim 3.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: time-check
  name: time-check
  namespace: dvl1987
spec:
  containers:
  - command:
    - /bin/sh
    - -c
    - while true; do date; sleep $TIME_FREQ; done > /opt/time/time-check.log
    image: busybox
    name: time-check
    env:
    - name: TIME_FREQ
      valueFrom:
        configMapKeyRef:
          name: time-config
          key: TIME_FREQ
    volumeMounts:
    - name: log
      mountPath: /opt/time
  volumes:
    - name: log
      emptyDir: {}
$ kubectl apply -f 3.yaml
pod/time-check created
$ kubectl exec -it -n dvl1987 time-check -- tail -n1 /opt/time/time-check.log
Wed Feb 18 13:12:45 UTC 2026

$ kubectl create deploy nginx-deploy --image nginx:1.16 --replicas 4 --dry-run=client -oyaml > 4.yaml
$ vim 4.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx-deploy
  name: nginx-deploy
spec:
  replicas: 4
  selector:
    matchLabels:
      app: nginx-deploy
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 2
  template:
    metadata:
      labels:
        app: nginx-deploy
    spec:
      containers:
      - image: nginx:1.16
        name: nginx
$ kubectl apply -f 4.yaml
deployment.apps/nginx-deploy created
$ kubectl set image deploy nginx-deploy nginx=nginx:1.17
deployment.apps/nginx-deploy image updated
$ kubectl rollout undo deploy nginx-deploy
deployment.apps/nginx-deploy rolled back

$ kubectl create deploy redis --image redis:alpine --replicas 1 --port 6379 --dry-run=client -oyaml > 5.yaml
$ vim 5.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: redis
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - image: redis:alpine
        name: redis
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: 200m
        volumeMounts:
          - name: data
            mountPath: /redis-master-data
          - name: redis-config
            mountPath: /redis-master
      volumes:
        - name: data
          emptyDir: {}
        - name: redis-config
          configMap:
            name: redis-config
$ kubectl apply -f 5.yaml
deployment.apps/redis created
$ kubectl exec -it redis-5c768c8998-jlcdk -- cat /redis-master/redis-config
maxmemory 2mb
maxmemory-policy allkeys-lru
```

</details>

<details>
<summary>
Lab: Lightning Lab - 2
</summary>

```shell
$ kubectl get po -n dev1401 nginx1401 -oyaml | sed '/status:/,$d' | sed 's/8080/9080/g' | sed '/timeoutSeconds/a\    livenessProbe:\n      exec:\n        command: ["ls", "/var/www/html/file_check"]\n      initialDelaySeconds: 10\n      periodSeconds: 60' | kubectl replace --force -f -
pod "nginx1401" deleted from dev1401 namespace
pod/nginx1401 replaced
$ kubectl get po -n dev1401 nginx1401
NAME        READY   STATUS    RESTARTS   AGE
nginx1401   1/1     Running   0          25s

$ kubectl create cronjob dice --image kodekloud/throw-dice --schedule "* * * * *" --dry-run=client -oyaml > 2.yaml
$ kubectl create cj dice --image kodekloud/throw-dice --schedule "* * * * *" --dry-run=client -oyaml > 2.yaml
$ vim 2.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dice
spec:
  jobTemplate:
    metadata:
      name: dice
    spec:
      template:
        spec:
          containers:
          - image: kodekloud/throw-dice
            name: dice
          restartPolicy: OnFailure
      backoffLimit: 25
      activeDeadlineSeconds: 20
  schedule: '* * * * *'
$ kubectl apply -f 2.yaml
cronjob.batch/dice created
$ kubectl get cj
NAME   SCHEDULE    TIMEZONE   SUSPEND   ACTIVE   LAST SCHEDULE   AGE
dice   * * * * *   <none>     False     0        25s             69s

$ kubectl run my-busybox -n dev2406 --image busybox --command --dry-run=client -oyaml -- sh -c 'sleep 3600' > 3.yaml
$ kubectl describe node controlplane | grep -i labels -A3
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=controlplane
$ vim 3.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: my-busybox
  name: my-busybox
  namespace: dev2406
spec:
  nodeSelector:
    kubernetes.io/hostname: controlplane
  containers:
  - command:
    - sh
    - -c
    - sleep 3600
    image: busybox
    name: secret
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secret-volume
      readOnly: true
  volumes:
    - name: secret-volume
      secret:
        secretName: dotfile-secret
$ kubectl apply -f 3.yaml
pod/my-busybox created
$ kubectl get po -n dev2406 my-busybox -owide
NAME         READY   STATUS    RESTARTS   AGE   IP            NODE           NOMINATED NODE   READINESS GATES
my-busybox   1/1     Running   0          37s   172.17.0.10   controlplane   <none>           <none>

$ kubectl get svc -A | grep 30093
ingress-nginx   ingress-nginx-controller             NodePort    172.20.68.39    <none>        80:30093/TCP,443:31870/TCP   18m
$ kubectl get ingressclass
NAME    CONTROLLER             PARAMETERS   AGE
nginx   k8s.io/ingress-nginx   <none>       25m
$ kubectl get svc -A | grep video
default         video-service                        ClusterIP   172.20.33.91    <none>        8080/TCP                     6m43s
$ kubectl get svc -A | grep apparel
default         apparels-service                     ClusterIP   172.20.50.220   <none>        8080/TCP                     6m51s
$ kubectl create ing ingress-vh-routing --class nginx --rule "watch.ecom-store.com/video=video-service:8080" --rule "apparels.ecom-store.com/wear=apparels-service:8080" --annotation nginx.ingress.kubernetes.io/rewrite-target=/
ingress.networking.k8s.io/ingress-vh-routing created
$ curl http://watch.ecom-store.com:30093/video | head -n2
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   293  100   293    0     0  84002      0 --:--:-- --:--:-- --:--:-- 97666
<!doctype html>
<title>Hello from Flask</title>
$ curl http://apparels.ecom-store.com:30093/wear | head -n2
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   296  100   296    0     0  96260      0 --:--:-- --:--:-- --:--:--  144k
<!doctype html>
<title>Hello from Flask</title>

$ kubectl logs dev-pod-dind-878516 -c log-x | grep WARNING > /opt/dind-878516_logs.txt
```

</details>

## Mock Exams

<details>
<summary>
Mock Exam - 1
</summary>

```shell
$ kubectl run nginx-448839 --image nginx:alpine
pod/nginx-448839 created
$ kubectl get po nginx-448839
NAME           READY   STATUS    RESTARTS   AGE
nginx-448839   1/1     Running   0          29s

$ kubectl create ns apx-z993845
namespace/apx-z993845 created
$ kubectl get ns apx-z993845
NAME          STATUS   AGE
apx-z993845   Active   18s

$ kubectl create deploy httpd-frontend --image httpd:2.4-alpine --replicas 3
deployment.apps/httpd-frontend created
$ kubectl get po | grep frontend
httpd-frontend-7dd67f597c-kc4hs   1/1     Running   0          11s
httpd-frontend-7dd67f597c-lq4pj   1/1     Running   0          11s
httpd-frontend-7dd67f597c-zjnsd   1/1     Running   0          11s

$ kubectl run messaging --image redis:alpine -l tier=msg
pod/messaging created
$ kubectl get po -l tier=msg
NAME        READY   STATUS    RESTARTS   AGE
messaging   1/1     Running   0          11s

$ kubectl get po | grep rs-
rs-d33393-67zwm                   0/1     InvalidImageName   0          99s
rs-d33393-chlkh                   0/1     InvalidImageName   0          99s
rs-d33393-ls2gp                   0/1     InvalidImageName   0          99s
rs-d33393-rk86x                   0/1     InvalidImageName   0          99s
$ kubectl set image rs rs-d33393 busybox-container=busybox
replicaset.apps/rs-d33393 image updated
$ kubectl scale rs rs-d33393 --replicas 0
replicaset.apps/rs-d33393 scaled
$ kubectl scale rs rs-d33393 --replicas 4
replicaset.apps/rs-d33393 scaled
$ kubectl get po | grep rs-
rs-d33393-4mk6f                   1/1     Running   0          18s
rs-d33393-6qphb                   1/1     Running   0          18s
rs-d33393-hvfjg                   1/1     Running   0          18s
rs-d33393-thh5m                   1/1     Running   0          18s

$ kubectl expose -n marketing deploy redis --name messaging-service --port 6379
service/messaging-service exposed
$ kubectl get svc -n marketing
NAME                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
messaging-service   ClusterIP   172.20.105.222   <none>        6379/TCP   10s

$ kubectl get po webapp-color -oyaml | sed '/status:/,$d' | sed 's/value: pink/value: green/g' | kubectl replace --force -f -
pod "webapp-color" deleted from default namespace
pod/webapp-color replaced
$ kubectl get po webapp-color
NAME           READY   STATUS    RESTARTS   AGE
webapp-color   1/1     Running   0          68s

$ kubectl create cm cm-3392845 --from-literal DB_NAME=SQL3322 --from-literal DB_HOST=sql322.mycompany.com --from-literal DB_PORT=3306
configmap/cm-3392845 created
$ kubectl get cm cm-3392845
NAME         DATA   AGE
cm-3392845   3      25s

$ kubectl create secret generic db-secret-xxdf --from-literal DB_Host=sql01 --from-literal DB_User=root --from-literal DB_Password=password123
secret/db-secret-xxdf created
$ kubectl get secret db-secret-xxdf
NAME             TYPE     DATA   AGE
db-secret-xxdf   Opaque   3      5s

$ kubectl get po app-sec-kff3345 -oyaml | sed '/status:/,$d' > 10.yaml
$ vim 10.yaml
    securityContext:
      capabilities:
        add: ["SYS_TIME"]
$ kubectl replace --force -f 10.yaml
pod "app-sec-kff3345" deleted from default namespace
pod/app-sec-kff3345 replaced
$ kubectl get po app-sec-kff3345
NAME              READY   STATUS    RESTARTS   AGE
app-sec-kff3345   1/1     Running   0          20s

$ kubectl get po -A | grep e-com-1123
e-commerce    e-com-1123                                 1/1     Running   0          18m
$ kubectl logs -n e-commerce e-com-1123 > /opt/outputs/e-com-1123.logs

$ vim 12.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-analytics
spec:
  capacity:
    storage: 100Mi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "/pv/data-analytics"
$ kubectl apply -f 12.yaml
persistentvolume/pv-analytics created
$ kubectl get pv pv-analytics
NAME           CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS      CLAIM   STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pv-analytics   100Mi      RWX            Retain           Available                          <unset>                          34s

$ kubectl create deploy redis --image redis:alpine
deployment.apps/redis created
$ kubectl expose deploy redis --name redis --port 6379
service/redis exposed
$ vim 13.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: redis-access
spec:
  podSelector:
    matchLabels:
      app: redis
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          access: redis
    ports:
    - protocol: TCP
      port: 6379
$ kubectl apply -f 13.yaml
networkpolicy.networking.k8s.io/redis-access created
$ kubectl run test -it --rm --image busybox --command -- sh -c "nc -vz -w 5 redis:6379; sleep 5"
nc: redis:6379 (172.20.50.1:6379): Connection timed out
pod "test" deleted from default namespace
$ kubectl run test -it --rm --image busybox -l access=redis --command -- sh -c "nc -vz -w 5 redis:6379; sleep 5"
redis:6379 (172.20.50.1:6379) open
pod "test" deleted from default namespace

$ kubectl run sega --image busybox --command --dry-run=client -oyaml -- sh -c "sleep 3600" > 14.yaml
$ vim 14.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: sega
  name: sega
spec:
  containers:
  - command:
    - sh
    - -c
    - sleep 3600
    image: busybox
    name: tails
  - image: nginx
    name: sonic
    env:
    - name: NGINX_PORT
      value: "8080"
$ kubectl apply -f 14.yaml
pod/sega created
$ kubectl get po sega
NAME   READY   STATUS    RESTARTS   AGE
sega   2/2     Running   0          13s
```

</details>

<details>
<summary>
Mock Exam - 2
</summary>

```shell
$ kubectl create deploy my-webapp --image nginx --replicas 2 --dry-run=client -oyaml | sed '5a\    tier: frontend' | kubectl apply -f -
deployment.apps/my-webapp created
$ kubectl expose deploy my-webapp --name front-end-service --port 80 --type NodePort --dry-run=client -oyaml | sed '/targetPort:/a\    nodePort: 30083' | kubectl apply -f -
service/front-end-service created
$ curl http://localhost:30083 | head -n4
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   615  100   615    0     0   935k      0 --:--:-- --:--:-- --:--:--  600k
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>

$ kubectl taint node node01 app_type=alpha:NoSchedule
node/node01 tainted
$ kubectl run alpha --image redis --dry-run=client -oyaml > 2.yaml
$ vim 2.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: alpha
  name: alpha
spec:
  containers:
  - image: redis
    name: alpha
  tolerations:
  - key: "app_type"
    operator: "Equal"
    value: "alpha"
    effect: "NoSchedule"
$ kubectl apply -f 2.yaml
pod/alpha created
$ kubectl get po alpha  -owide
NAME    READY   STATUS    RESTARTS   AGE   IP           NODE     NOMINATED NODE   READINESS GATES
alpha   1/1     Running   0          10s   172.17.1.6   node01   <none>           <none>

$ kubectl label node controlplane app_type=beta
node/controlplane labeled
$ kubectl create deploy beta-apps --image nginx --replicas 3 --dry-run=client -oyaml > 3.yaml
$ vim 3.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: beta-apps
  name: beta-apps
spec:
  replicas: 3
  selector:
    matchLabels:
      app: beta-apps
  template:
    metadata:
      labels:
        app: beta-apps
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: app_type
                operator: In
                values:
                - beta
      containers:
      - image: nginx
        name: nginx
$ kubectl apply -f 3.yaml
deployment.apps/beta-apps created
$ kubectl get po -owide | grep beta
beta-apps-68c7d97b86-cpp6k   1/1     Running   0          22s    172.17.0.7   controlplane   <none>           <none>
beta-apps-68c7d97b86-pt4bv   1/1     Running   0          22s    172.17.0.8   controlplane   <none>           <none>
beta-apps-68c7d97b86-zrnc6   1/1     Running   0          23s    172.17.0.6   controlplane   <none>           <none>

$ kubectl get svc my-video-service
NAME               TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
my-video-service   ClusterIP   172.20.225.16   <none>        8080/TCP   70s
$ kubectl create ing video --rule 'ckad-mock-exam-solution.com/video=my-video-service:8080' --annotation nginx.ingress.kubernetes.io/rewrite-target=/
ingress.networking.k8s.io/video created
$ curl http://ckad-mock-exam-solution.com:30093/video | head -n2
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   293  100   293    0     0  82257      0 --:--:-- --:--:-- --:--:-- 97666
<!doctype html>
<title>Hello from Flask</title>

$ kubectl get po pod-with-rprobe -oyaml | sed '/status:/,$d' | sed '/imagePullPolicy/a\    readinessProbe:\n      httpGet:\n        path: /ready\n        port: 8080' | kubectl replace --force -f -
pod "pod-with-rprobe" deleted from default namespace
pod/pod-with-rprobe replaced

$ kubectl run nginx1401 --image nginx --dry-run=client -oyaml | sed '10a\    livenessProbe:\n      exec:\n        command: ["ls", "/var/www/html/probe"]\n      initialDelaySeconds: 10\n      periodSeconds: 60' | kubectl apply -f -
pod/nginx1401 created
$ kubectl get po nginx1401
NAME        READY   STATUS    RESTARTS   AGE
nginx1401   1/1     Running   0          19s

$ kubectl create job whalesay --image busybox --dry-run=client -oyaml -- sh -c 'echo "cowsay I am going to ace CKAD!"' > 7.yaml
$ vim 7.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: whalesay
spec:
  template:
    metadata: {}
    spec:
      containers:
      - command:
        - sh
        - -c
        - echo "cowsay I am going to ace CKAD!"
        image: busybox
        name: whalesay
      restartPolicy: Never
  completions: 10
  backoffLimit: 6
$ kubectl apply -f 7.yaml
job.batch/whalesay created

$ kubectl run multi-pod --image busybox --command --dry-run=client -oyaml -- sh -c 'sleep 4800' > 8.yaml
$ vim 8.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    run: multi-pod
  name: multi-pod
spec:
  containers:
  - image: nginx
    name: jupiter
    env:
    - name: type
      value: planet
  - command:
    - sh
    - -c
    - sleep 4800
    image: busybox
    name: europa
    env:
    - name: type
      value: moon
$ kubectl apply -f 8.yaml
pod/multi-pod created

$ vim 9.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: custom-volume
spec:
  capacity:
    storage: 50Mi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: "/opt/data"
  persistentVolumeReclaimPolicy: Retain
$ kubectl apply -f 9.yaml
persistentvolume/custom-volume created
```

</details>

---

# Retrospective

시험에 접속하면 본인 확인 및 부정행위 방지를 위한 몇가지 절차를 수행한다.
이때 나는 고양이를 키워 문제가 될 수 있는 상황이였고 시험을 보며 고양이 눈치도 살폈다... (🐈‍⬛)

본 시험에서는 17문제가 출제되었고 2시간 중 1시간 30분 정도에 다 풀었어서 그대로 제출했었다.
오히려 CKA에서는 ETCD 백업이나 클러스터 전반적인 장애 복구를 위주로 했다보니 context를 변경하며 실수하지 않도록 유의했지만, CKAD에서는 `ssh`로 각 서버 접속 후 클러스터를 조작해서 오히려 훨씬 쉬웠다.
그 외에는 [`ResourceQuota`](https://kubernetes.io/docs/concepts/policy/resource-quotas/)와 [`LimitRange`](https://kubernetes.io/docs/concepts/policy/limit-range/)가 KodeKloud 예상 문제에 없었는데 2문제 출제되어 좀 당황했지만 공식 문서를 읽으면서 차분히 수행했기에 좋은 결과가 있었던 것 같다.

~~[Kubestronauts](https://www.cncf.io/training/kubestronaut/)를 하고는 싶지만,,, 자격증 가격 및 환율이 너무 올라서 미련 없이 Kubernetes 자격증 시리즈는 끝낼 수 있을 것 같다...~~
