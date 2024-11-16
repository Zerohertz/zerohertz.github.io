---
title: 'Home Server: Cloud'
date: 2023-08-22 23:53:49
categories:
- 0. Daily
tags:
- Home Server
- Kubernetes
- Traefik
---
# Introduction

[Dropbox](https://www.dropbox.com)와 같은 cloud storage service를 Kubernetes로 배포하기 위해 [Nextcloud](https://github.com/nextcloud)을 사용한다!

![Nextcloud](/images/home-server-cloud/262398322-65cec9f6-d359-422a-8735-e5db355e4fb7.png)

<!-- More -->

---

# Setup

## Storage

Cloud "storage"이기 때문에 K8s에서 사용할 `StorageClass`, `PersistentVolume`, `PersistentVolumeClaim`을 모두 정의해줘야한다.

+ `nextcloud-pvc`: Nextcloud에서 설정하는 여러 파일들이 저장될 곳
+ `nextcloud-data-pvc`: 사용자가 사용할 저장공간

```yaml storage.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nextcloud-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nextcloud-pv
spec:
  storageClassName: nextcloud
  accessModes:
    - ReadWriteOnce
  capacity:
    storage: 10Gi
  hostPath:
    path: "/${PATH}/nextcloud/etc"
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nextcloud-data-pv
spec:
  storageClassName: nextcloud
  accessModes:
    - ReadWriteOnce
  capacity:
    storage: 600Gi
  hostPath:
    path: "/${PATH}/nextcloud/data"
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nextcloud-pvc
  namespace: nextcloud
spec:
  storageClassName: nextcloud
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nextcloud-data-pvc
  namespace: nextcloud
spec:
  storageClassName: nextcloud
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 600Gi
```

## Helm

Nextcloud는 아래와 같이 Helm으로 간편히 수정 및 배포할 수 있다.
`helm show values`로 설정할 수 있는 값들을 살펴볼 수 있다.

```shell
$ helm repo add nextcloud https://nextcloud.github.io/helm/
$ helm repo update
$ helm show values nextcloud/nextcloud --version 3.5.22 > nextcloud-values.yaml
```

Traefik을 통해 HTTPS로 배포할 예정이라서 `phpClientHttpsFix`을 설정했고, 앞서 정의한 볼륨들을 `StorageClass`와 `PVC`의 이름에 맞게 설정한다.

```yaml values.yaml
nextcloud:
  username: ${USER}
  password: ${PASSWORD}
phpClientHttpsFix:
  enabled: true
  protocol: https
persistence:
  enabled: true
  storageClass: nextcloud-storage
  existingClaim: nextcloud-pvc
  accessMode: ReadWriteOnce
  size: 10Gi
  nextcloudData:
    enabled: true
    storageClass: nextcloud-storage
    existingClaim: nextcloud-data-pvc
    accessMode: ReadWriteOnce
    size: 600Gi
```

## Traefik

Ingress는 service port인 `8080`을 `websecure`로 정의했다.

```yaml traefik.yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: nextcloud
  namespace: nextcloud
spec:
  entryPoints:
  - websecure
  routes:
  - match: Host(`cloud.${DDNS}`)
    kind: Rule
    services:
    - name: nextcloud
      port: 8080
  tls:
    certResolver: ${RESOLVER}
```

## K8s

최종적으로 배포는 아래와 같이 진행하면 된다.

```shell
$ kubectl create namespace nextcloud
$ kubectl apply -f storage.yaml
$ helm upgrade --install nextcloud nextcloud/nextcloud -f values.yaml --namespace nextcloud --version 3.5.22
$ kubectl apply -f traefik.yaml
```

`docker exec` 혹은 `PVC`에서 지정한 경로로 이동해 아래 파일과 같이 본인의 DNS 혹은 DDNS를 신뢰할 수 있도록 정의하면 끝이다.

```php etc/config/config.php
...
  array (
    0 => 'localhost',
    1 => 'nextcloud.kube.home',
    2 => 'cloud.${DDNS}',
  ),
...
```

---

# Hands-on

생각보다 편하게 잘 되어있다.
윈도우, 맥 그리고 아이폰 환경에서 모두 앱을 지원하고 그것을 통해 기존 cloud와 같이 동기화 및 링크를 통한 파일 공유까지 모두 사용할 수 있다.
또한 보안을 위한 2FA 혹은 로그인 되어있는 환경의 승인 등 Nextcloud 내 다양한 앱을 쉽게 사용할 수 있다.
아래는 유선으로 연결된 home server가 열심히 파일들을 동기화하는 중인 모습을 캡쳐한 것이다.

![Grafana](/images/home-server-cloud/262403002-c6b91e98-4146-4e52-80f3-136a215cd204.png)

---

# PostgreSQL

따로 backend를 설정해주지 않으면 Nextcloud는 SQLite를 사용한다.
하지만 보안 및 성능 이슈가 있을 수 있으니 다른 DB를 사용하는 것을 권장하여 아래와 같은 설정을 진행했다.

```yaml values.yaml
postgresql:
  enabled: true
  global:
    postgresql:
      auth:
        username: ${USERNAME}
        password: ${PASSWORD}
        database: nextcloud
  primary:
    initContainers:
      - name: volume-permissions
        image: busybox
        command: ["sh", "-c", "chown -R 1001:100 /var/lib/postgresql/data"]
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
    persistence:
      enabled: true
      storageClass: nextcloud-storage
      existingClaim: nextcloud-backend-pvc
```

그리고 Nextcloud container에 접속해서 아래 명령어를 실행하고 DB의 비밀번호를 입력하면 끝이다.

```shell
$ docker exec -it -u www-data k8s_nextcloud_nextcloud-..._nextcloud_... bash
$ php occ db:convert-type --clear-schema pgsql ${USERNAME} ${HOST} ${DB_NAME}
```

~~는 삽질 5시간은 한듯...~~