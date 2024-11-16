---
title: 'Traefik: OAuth'
date: 2023-08-16 12:41:02
categories:
- 3. DevOps
tags:
- Kubernetes
- Traefik
- Home Server
---
# Google OAuth

[여기](https://console.cloud.google.com/)에 접속해 아래의 과정을 수행한다.

1. 프로젝트 생성
2. OAuth 동의 화면 생성
3. OAuth 클라이언트 ID 생성

![oauth](/images/traefik-oauth/oauth.png)

<!-- More -->

이렇게 진행하고 아래의 manifest를 Kubernetes에 적용하면 `https://${DDNS}/test` 접속 시 Google OAuth로 이동하고 `WHITELIST`에 지정된 사용자면 `https://${DDNS}/test`로 다시 이동하게 된다.

```yaml ingress-example-oauth-google.yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: nginx-service
  namespace: google-oauth
spec:
  entryPoints:
  - websecure
  routes:
  - match: Host(`${DDNS}`) && PathPrefix(`/test`)
    kind: Rule
    middlewares:
    - name: forward-auth
    - name: nginx-service-strip-mw
    services:
    - name: nginx-service
      port: 80
  tls:
    certResolver: ${RESOLVER}
---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: nginx-service-strip-mw
  namespace: google-oauth
spec:
  stripPrefix:
    prefixes:
    - "/test"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: forward-auth
  name: forward-auth
  namespace: google-oauth
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forward-auth
  template:
    metadata:
      labels:
        app: forward-auth
    spec:
      containers:
      - name: traefik-forward-auth
        image: thomseddon/traefik-forward-auth:2
        imagePullPolicy: Always
        env:
        - name: PROVIDERS_GOOGLE_CLIENT_ID
          value: ${CLIENT_ID}
        - name: PROVIDERS_GOOGLE_CLIENT_SECRET
          value: ${CLIETN_SECRET}
        - name: SECRET
          value: ${SECRET}
        - name: COOKIE_DOMAIN
          value: ${DDNS}
        - name: AUTH_HOST
          value: ${DDNS}
        - name: INSECURE_COOKIE
          value: "false"
        - name: URL_PATH
          value: "/test/_oauth"
        - name: WHITELIST
          value: ${EMAIL}
        - name: LOG_LEVEL
          value: debug
        ports:
        - containerPort: 4181
---
apiVersion: v1
kind: Service
metadata:
  name: forward-auth
  namespace: google-oauth
  labels:
    app: traefik
spec:
  type: ClusterIP
  selector:
    app: forward-auth
  ports:
  - name: auth-http
    port: 4181
    targetPort: 4181
---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: forward-auth
  namespace: google-oauth
spec:
  forwardAuth:
    address: http://forward-auth.tmp.svc.cluster.local:4181
    authResponseHeaders:
    - X-Forwarded-User
```

다른 서비스에도 아래와 같이 잘 적용된다.

![grafana](/images/traefik-oauth/grafana.gif)

만약 subdomain을 사용한다면 `svc.${DDNS}`에 접속 시 `auth.${DDNS}`에 인증을 거치도록 아래와 같이 구성할 수 있다.

```yaml oauth.yaml
kind: Secret
apiVersion: v1
metadata:
  name: traefik-forward-auth-secrets
  namespace: oauth
  labels:
    name: traefik
type: Opaque
data:
  PROVIDERS_GOOGLE_CLIENT_ID:
  PROVIDERS_GOOGLE_CLIENT_SECRET:
  SECRET:
  # auth.${DDNS}
  AUTH_HOST:
  # /_oauth
  URL_PATH:
  # ${EMAIL}
  WHITELIST:
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: forward-auth
  name: forward-auth
  namespace: oauth
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forward-auth
  template:
    metadata:
      labels:
        app: forward-auth
    spec:
      containers:
      - name: traefik-forward-auth
        image: thomseddon/traefik-forward-auth:2
        imagePullPolicy: Always
        env:
        - name: PROVIDERS_GOOGLE_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: traefik-forward-auth-secrets
              key: PROVIDERS_GOOGLE_CLIENT_ID
        - name: PROVIDERS_GOOGLE_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: traefik-forward-auth-secrets
              key: PROVIDERS_GOOGLE_CLIENT_SECRET
        - name: SECRET
          valueFrom:
            secretKeyRef:
              name: traefik-forward-auth-secrets
              key: SECRET
        - name: AUTH_HOST
          valueFrom:
            secretKeyRef:
              name: traefik-forward-auth-secrets
              key: AUTH_HOST
        - name: INSECURE_COOKIE
          value: "false"
        - name: URL_PATH
          valueFrom:
            secretKeyRef:
              name: traefik-forward-auth-secrets
              key: URL_PATH
        - name: WHITELIST
          valueFrom:
            secretKeyRef:
              name: traefik-forward-auth-secrets
              key: WHITELIST
        - name: LOG_LEVEL
          value: debug
        ports:
        - containerPort: 4181
---
apiVersion: v1
kind: Service
metadata:
  name: forward-auth
  namespace: oauth
  labels:
    app: traefik
spec:
  type: ClusterIP
  selector:
    app: forward-auth
  ports:
  - name: auth-http
    port: 4181
    targetPort: 4181
---
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: auth-ingress-route
  namespace: oauth
spec:
  entryPoints:
  - web
  routes:
  - match: Host(`auth.${DDNS}`)
    kind: Rule
    services:
    - name: forward-auth
      port: 4181
```

---

# GitHub OAuth

동일한 도메인에 대해 여러 서비스를 배포하고 사용하니 어려운게 참 많아서 GitHub OAuth는 포기...

```yaml ingress-example-oauth-github.yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: nginx-service
  namespace: github-oauth
spec:
  entryPoints:
  - websecure
  routes:
  - match: Host(`${DDNS}`)
    kind: Rule
    middlewares:
    - name: forward-auth
    services:
    - name: nginx-service
      port: 80
  tls:
    certResolver: ${RESOLVER}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: forward-auth
  name: forward-auth
  namespace: github-oauth
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forward-auth
  template:
    metadata:
      labels:
        app: forward-auth
    spec:
      containers:
        - name: traefik-forward-auth
          image: thomseddon/traefik-forward-auth:2
          ports:
            - containerPort: 4181
              protocol: TCP
        env:
        - name: DEFAULT_PROVIDER
          value: generic-oauth
        - name: PROVIDERS_GENERIC_OAUTH_AUTH_URL
          value: https://github.com/login/oauth/authorize
        - name: PROVIDERS_GENERIC_OAUTH_TOKEN_URL
          value: https://github.com/login/oauth/access_token
        - name: PROVIDERS_GENERIC_OAUTH_USER_URL
          value: https://api.github.com/user
        - name: PROVIDERS_GENERIC_OAUTH_CLIENT_ID
          value: ${CLIENT_ID}
        - name: PROVIDERS_GENERIC_OAUTH_CLIENT_SECRET
          value: ${CLIENT_SECRET}
        - name: SECRET
          value: ${SECRET}
        - name: COOKIE_DOMAIN
          value: ${DDNS}
        - name: AUTH_HOST
          value: ${DDNS}
        - name: INSECURE_COOKIE
          value: "false"
        - name: URL_PATH
          value: "/_oauth"
        - name: WHITELIST
          value: ${EMAIL}
        - name: LOG_LEVEL
          value: debug
        livenessProbe:
          tcpSocket:
            port: 4181
          initialDelaySeconds: 20
          failureThreshold: 3
          successThreshold: 1
          periodSeconds: 10
          timeoutSeconds: 2
---
apiVersion: v1
kind: Service
metadata:
  name: forward-auth
  namespace: github-oauth
  labels:
    app: traefik
spec:
  type: ClusterIP
  selector:
    app: forward-auth
  ports:
  - name: auth-http
    port: 4181
    targetPort: 4181
---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: forward-auth
  namespace: github-oauth
spec:
  forwardAuth:
    address: http://forward-auth.github-oauth.svc.cluster.local:4181
    authResponseHeaders:
      - X-Forwarded-User
```