---
title: 'Kubernetes: Ingress'
date: 2023-08-08 23:43:40
categories:
- 3. DevOps
tags:
- Kubernetes
- Traefik
- Home Server
---
# Introduction

현대의 마이크로서비스 환경에서 쿠버네티스 클러스터 내의 애플리케이션으로 트래픽을 관리하는 것은 필수 기술입니다.
쿠버네티스는 클러스터로의 외부 트래픽을 제어하고 라우팅하는 강력한 솔루션인 Ingress를 제공합니다.

~~GPT-4 좋네~~

> Ingress: Cluster 내 서비스로의 외부 액세스를 관리하는 Kubernetes의 API 객체

Kubernetes의 ingress는 host 이름과 경로를 포함한 일련의 규칙을 기반으로 HTTP와 HTTPS routing을 제공하여 client 요청을 적절한 backend service로 연결하는 장점이 있다.
Ingress의 주요 구성 요소는 아래와 같다.

1. Ingress Resource
   + Traffic routing 규칙 정의
   + Host, 경로 및 traffic 흐름을 제어하는 다양한 기타 매개변수 포함
2. Ingress Controller
   + Ingress 리소스의 규칙을 읽고 실제 traffic routing 처리
   + [NGINX Ingress Controller](https://github.com/kubernetes/ingress-nginx), [Traefik](https://github.com/traefik/traefik), ...
3. Backend Service
   + Cluster 내에서 요청을 처리하는 실제 service
   + Ingress resource에서 정의한 대로 다양한 경로와 도메인을 통해 노출 가능

또한 사용하는 이유는 아래와 같다.

1. 간소화된 routing: Routing 규칙을 한 곳에서 집중 관리하여 관리 및 유지 보수 명료화
2. SSL/TLS 종료: SSL/TLS 인증서를 한 곳에서 처리
3. 경로 기반 routing: URL 경로를 기반으로 요청을 다른 서비스로 연결함으로써 보다 체계적인 구조 구축
4. 비용 효율: Routing 규칙을 통합함으로써 여러 load balancer의 필요 감소

<!-- More -->

---

# NGINX Ingress Controller vs. Traefik

## NGINX Ingress Controller

> NGINX Ingress Controller: 인기 있는 web server NGINX 기반 ingress controller

+ 성능: NGINX의 강력한 성능
+ 확장성: 큰 규모의 어플리케이션에 적합
+ 사용자 정의: 많은 사용자 정의 옵션과 플러그인
+ 구성 복잡성: 처음 사용자에게는 설정이 복잡할 수 있음

## Traefik

> Traefik: 최신 microservice를 위해 특별히 설계된 dynamic reverse proxy ingress controller

+ 동적 구성: 마이크로서비스 환경에 적합한 동적 routing
+ 간단한 구성: 빠른 설정과 간단한 관리
+ Let's Encrypt 지원: 자동 SSL 인증서 관리
+ 미들웨어 지원: 다양한 미들웨어 옵션

## Conclusion

|Feature|NGINX Ingress Controller|Traefik|
|:-:|:-:|:-:|
|성능|높음|중간|
|확장성|높음|중간|
|설정 복잡성|복잡|단순|
|미들웨어 지원|제한적|넓은 범위|
|자동 SSL 인증서 관리|제한적|지원|
|커뮤니티 및 지원|넓고 강력|활발|

+ NGINX Ingress Controller는 성능과 확장성이 중요한 큰 규모의 프로젝트에 적합할 수 있습니다.
+ Traefik은 동적 마이크로서비스 환경과 빠른 개발이 필요한 프로젝트에 더 적합할 수 있습니다.

<details>
<summary>
그 외의 여러 Ingress Controller들
</summary>

1. HAProxy Ingress Controller: HAProxy를 기반으로 한 인기 있는 Ingress 컨트롤러로, 고성능 및 확장성에 중점을 둡니다.
2. Contour: Contour는 Envoy 프록시를 기반으로 하며, 직관적인 구성과 최신 Ingress 기능을 지원합니다.
3. Istio Ingress Gateway: Istio 서비스 메시와 함께 작동하는 Ingress 컨트롤러로, 트래픽 관리, 보안, 관찰 등을 제공합니다.
4. Kong Ingress Controller: Kong은 API 게이트웨이로도 사용되며, 풍부한 플러그인 생태계와 사용자 친화적인 구성을 제공합니다.
5. AWS ALB Ingress Controller: Amazon Web Services (AWS)의 Application Load Balancer (ALB)를 활용하는 Ingress 컨트롤러로, AWS 환경에서 자동 확장 및 탄력성을 제공합니다.
6. Gloo Edge: Solo.io에서 개발한 Gloo Edge는 Envoy를 기반으로 하며, 함수 레벨 라우팅, gRPC, 웹소켓 지원 등 다양한 기능을 제공합니다.
7. Azure Application Gateway Ingress Controller: Microsoft Azure의 Application Gateway를 활용한 Ingress 컨트롤러로, Azure 기반 클라우드 환경에서 최적화된 트래픽 라우팅을 지원합니다.

</details>

---

# Hands-on

Traefik을 통한 ingress를 알아보자!

## HTTP

```yaml ingress-example-http.yaml
apiVersion: v1
kind: Service
metadata:
  name: echo-server
  namespace: example
  labels:
    app: echo-server
spec:
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  selector:
    app: echo-server
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: echo-server
  namespace: example
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: echo-server
  template:
    metadata:
      labels:
        app: echo-server
    spec:
      containers:
        - name: echo-server
          image: ealen/echo-server:0.5.2
          imagePullPolicy: Always
          ports:
            - containerPort: 80
          resources:
            limits:
              cpu: "0.5"
              memory: "1Gi"
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: traefik-routers
  namespace: example
spec:
  ingressClassName: "traefik"
  rules:
      - http:
          paths:
            - path: /
              pathType: Prefix
              backend:
                service:
                  name: echo-server
                  port:
                    number: 80
```

```shell
$ kubectl create ns example
namespace/example created
$ kubectl apply -f ingress-example.yaml
service/echo-server created
deployment.apps/echo-server created
ingress.networking.k8s.io/traefik-routers created
$ kubectl get svc -A
NAMESPACE     NAME             TYPE           CLUSTER-IP      EXTERNAL-IP       PORT(S)                      AGE
...
example       echo-server      ClusterIP      10.43.60.106    <none>            80/TCP                       2m32s
kube-system   traefik          LoadBalancer   10.43.78.230    192.168.219.200   80:31024/TCP,443:31025/TCP   36h
```

만약 Traefik의 port을 원하는 port로 수정하고 싶다면 `kubectl edit svc traefik -n kube-system`을 실행하여 아래와 같이 수정하면 된다.

```yaml
spec:
  ...
  ports:
  - name: web
    nodePort: ${Web_Port}
    port: 80
    protocol: TCP
    targetPort: web
  - name: websecure
    nodePort: ${Websecure_Port}
    port: 443
    protocol: TCP
    targetPort: websecure
    ...
```

그러면 아래와 같이 잘 실행이 된다!

![http](/images/k8s-ingress/http.png)

## HTTPS

자체 서명된 인증서로 위의 service를 배포하면 아래와 같다.

```shell
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout tls.key -out tls.crt -subj "/CN=${HOST}"
$ kubectl create secret tls my-tls-cert --cert=tls.crt --key=tls.key -n example
```

```yaml ingress-example-https.yaml
...
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: traefik-routers
  namespace: example
spec:
  ingressClassName: "traefik"
  tls:
  - secretName: my-tls-cert
  ...
```

![https-1](/images/k8s-ingress/https-1.png)

위 사진에서 알 수 있듯, 자체 서명한 인증서를 사용했기 때문에 우측과 같은 경고문이 뜬다.
Traefik은 Let's Encrypt와 같은 ACME (Automated Certificate Management Environment) protocol을 지원하여 TLS 인증서의 자동 발급 및 갱신을 처리할 수 있다.
Traefik 구성에서 ACME 제공자를 활성화하고 필요한 인증 정보를 설정하면, Traefik은 지정된 domain에 대한 인증서를 자동으로 발급 받을 수 있으며, 인증서가 만료되기 전에 자동으로 갱신한다.
이러한 자동 인증서 관리 기능은 web service의 보안을 강화하고, 수동으로 인증서를 관리하는 복잡성을 줄여준다.
위 특징들을 적용하기 위해선 아래의 과정을 거치면 된다.

```shell
$ kubectl edit deployment traefik -n kube-system
```

```yaml
...
spec:
  ...
  template:
    ...
    spec:
      containers:
      - args:
        ...
        - --certificatesresolvers.myresolver.acme.email=${User_Mail}
        - --certificatesresolvers.myresolver.acme.storage=/data/acme.json
        - --certificatesresolvers.myresolver.acme.tlschallenge=true
        ...
```

```shell
$ kubectl rollout restart deployment traefik -n kube-system
```

```yaml ingress-example-https-cert.yaml
...
---
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: echo-server
  namespace: example
spec:
  entryPoints:
    - websecure
  routes:
  - match: Host(`${DDNS}`)
    kind: Rule
    services:
    - name: echo-server
      port: 80
  tls:
    certResolver: myresolver
```

```shell
$ kubectl apply -f ingress-example-https-cert.yaml
```

![https-2](/images/k8s-ingress/https-2.png)

성공 !!
만약 path를 변경하고 싶다면 `spec.routes.match`를 ``Host(`${DDNS}`) && PathPrefix(`/${PATH}`)``와 같이 지정해주면 된다.

[이런 사이트](https://www.ssllabs.com/ssltest/analyze.html)를 이용하면 아래와 같이 서버의 보안을 평가 받을 수 있다.

![ssltest](/images/k8s-ingress/ssltest.png)

---

# Traefik Dashboard

```shell
$ echo -n "${ID}:$(openssl passwd -apr1 ${PASSWD})" | base64
```

위 명령어를 통해 encoding된 ID와 비밀번호를 얻을 수 있고, 여기서 출력된 값을 아래 `${Base64}`에 입력하면 접속 시 사용자에게 ID와 비밀번호를 요구하게된다.

```yaml traefik.yaml
apiVersion: v1
kind: Secret
metadata:
  name: dashboard-auth-secret
  namespace: monitoring
data:
  users: ${Base64}
---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: traefik-dashboard
  namespace: monitoring
spec:
  basicAuth:
    secret: dashboard-auth-secret
---
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: traefik-dashboard
  namespace: monitoring
spec:
  entryPoints:
    - websecure
  routes:
  - match: Host(`${DDNS}`)
    kind: Rule
    middlewares:
    - name: traefik-dashboard
    services:
    - name: api@internal
      kind: TraefikService
  tls:
    certResolver: myresolver
```

아래와 같이 접속하면 ID와 비밀번호를 묻고 로그인하면 Traefik의 dashboard를 확인할 수 있다!

![traefik-dashboard](/images/k8s-ingress/traefik-dashboard.png)