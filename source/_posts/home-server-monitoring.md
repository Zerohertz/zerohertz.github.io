---
title: 'Home Server: Monitoring'
date: 2023-08-07 22:09:42
categories:
- 3. DevOps
tags:
- Home Server
- Kubernetes
- Grafana
- Prometheus
- Traefik
---
# Introduction

Node exporter의 system metric들을 Prometheus로 수집하고 Grafana로 시각화!

![Monitoring](/images/home-server-monitoring/258835852-b2f1349a-c4e5-4c95-8492-c5033f5f3980.gif)

<!-- More -->

---

# Node Exporter

```shell
$ wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
$ tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz
$ sudo mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/node_exporter
```

```yaml /etc/systemd/system/node_exporter.service
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=root
Group=root
Type=simple
ExecStart=/usr/local/bin/node_exporter # --web.listen-address=:9101 --collector.systemd

[Install]
WantedBy=multi-user.target
```

```shell
$ sudo systemctl daemon-reload
$ sudo systemctl start node_exporter
$ sudo systemctl enable node_exporter
```

`http://${CLUSTER_IP}:9100/metrics`에서 아래와 같이 log들을 확인할 수 있다.

![node_exporter](/images/home-server-monitoring/258650267-88fe8ad9-4552-4707-b643-5a1d74b12c6b.gif)

---

# Prometheus

```yaml configmap/prometheus.yaml
global:
  scrape_interval:     5s
  evaluation_interval: 5s
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
    - targets: ['localhost:9090']
  - job_name: 'node'
    static_configs:
    - targets: ['${CLUSTER_IP}:9101']
```

```shell
$ kubectl create namespace monitoring
$ kubectl create configmap prometheus --from-file=prometheus.yml=configmap/prometheus.yaml -n monitoring
```

```yaml prometheus.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: prometheus-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: prometheus-logs
  labels:
    type: prometheus-logs
spec:
  storageClassName: prometheus-storage
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "${DIRECTORY}/logs"
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus
  namespace: monitoring
spec:
  storageClassName: prometheus-storage
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  selector:
    matchLabels:
      type: prometheus-logs
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus
          ports:
            - containerPort: 9090
          volumeMounts:
          - name: data
            mountPath: /prometheus
          - name: config
            mountPath: /etc/prometheus/prometheus.yml
            subPath: prometheus.yml
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: prometheus
      - name: config
        configMap:
          name: prometheus
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  # type: NodePort
  ports:
    - port: 9090
      # nodePort: ${NODE_PORT}
  selector:
    app: prometheus
```

```shell
$ kubectl apply -f prometheus.yaml
```

![Prometheus](/images/home-server-monitoring/258821180-bfaf0f4f-9320-4597-bbc8-1d7f8485c49d.png)
![node_cpu](/images/home-server-monitoring/258826658-ad837d6a-eed1-4c3b-adb0-020ae0a8b511.png)

---

# Grafana

```yaml grafana.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
        - image: grafana/grafana
          name: grafana
          env:
            - name: GF_SECURITY_ADMIN_USER
              value: ${ID}
            - name: GF_SECURITY_ADMIN_PASSWORD
              value: ${PASSWORD}
          ports:
            - containerPort: 3000
          volumeMounts:
          - name: data
            mountPath: /var/lib/grafana
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: grafana
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  type: NodePort
  ports:
    - port: ${Grafana_Port}
      targetPort: 3000
      nodePort: ${Grafana_Port}
  selector:
    app: grafana
```

```bash
kubectl apply -f grafana.yaml
```

[Node Exporter Full](https://grafana.com/grafana/dashboards/1860-node-exporter-full/)를 import하여 dashboard를 구성했다!

![Grafana](/images/home-server-monitoring/258822274-e70803d6-19e8-4626-b369-4f8a1a9d584f.png)
![Monitoring_2](/images/home-server-monitoring/258820862-78d15f77-7366-4e91-a4f4-ad638b54df77.png)

---

# Ingress

<details>
<summary>
Ingress 시도 및 실패,,, (<code>traefik</code>)
</summary>

```yaml traefik-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: traefik-conf
  namespace: kube-system
data:
  traefik.toml: |
    [entryPoints]
      [entryPoints.web]
        address = ":80"
      [entryPoints.websecure]
        address = ":443"
      [entryPoints.grafana]
        address = ":${Grafana_Port}"
      [entryPoints.prometheus]
        address = ":${Prometheus_Port}"
```

```shell
$ kubectlapply -f traefik-deployment.yaml
configmap/traefik-conf created
$ kubectl edit deployment traefik -n kube-system
deployment.apps/traefik edited
```

```yaml traefik
...
spec:
  ...
  template:
    ...
    spec:
      containers:
        ...
        volumeMounts:
        - mountPath: /data
          name: data
        - mountPath: /tmp
          name: tmp
        - name: config
          mountPath: /etc/traefik
          readOnly: true
          ...
      volumes:
      - emptyDir: {}
        name: data
      - emptyDir: {}
        name: tmp
      - name: config
        configMap:
          name: traefik-conf
```

```shell
$ kubectl rollout restart deployment/traefik -n kube-system
deployment.apps/traefik restarted
$ kubectl edit svc traefik -n kube-system
```

```yaml traefik
spec:
  ...
  ports:
  ...
  - name: grafana
    nodePort: ${Grafana_Port}
    port: ${Grafana_Port}
    protocol: TCP
    targetPort: ${Grafana_Port}
  - name: prometheus
    nodePort: ${Prometheus_Port}
    port: ${Prometheus_Port}
    protocol: TCP
    targetPort: ${Prometheus_Port}
...
```

```bash
kubectlget svc -n kube-system
NAME             TYPE           CLUSTER-IP      EXTERNAL-IP       PORT(S)                                                      AGE
kube-dns         ClusterIP      10.43.0.10      <none>            53/UDP,53/TCP,9153/TCP                                       11m
metrics-server   ClusterIP      10.43.195.230   <none>            443/TCP                                                      11m
traefik          LoadBalancer   10.43.66.36     ${IP}             80:32633/TCP,443:31858/TCP,${Grafana_Port}:${Grafana_Port}/TCP,${Prometheus_Port}:${Prometheus_Port}/TCP   11m
```

</details>

<details>
<summary>
정신 못차리고 <code>HTTPS</code>를 도입해보려고 Ingress를 시도하려다 또 실패 ~
</summary>

```shell
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout tls.key -out tls.crt -subj "/CN=${HOST}" 
$ kubectl create secret tls grafana-tls --key tls.key --cert tls.crt -n monitoring
$ kubectl get secret -n monitoring
NAME          TYPE                DATA   AGE
grafana-tls   kubernetes.io/tls   2      36s
```

```yaml grafana.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grafana-ingress
  namespace: monitoring
spec:
  rules:
  - host: ${HOST}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: ${Grafana_Port}
  tls:
  - hosts:
    - ${HOST}
    secretName: grafana-tls
```

```shell
$ kubectl apply -f grafana.yaml
```

```yaml traefik.yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: grafana-ingressroute
  namespace: monitoring
spec:
  entryPoints:
    - websecure
  routes:
  - match: Host(`${HOST}`)
    kind: Rule
    services:
    - name: grafana
      port: ${Grafana_Port}
  tls:
    secretName: grafana-tls
```

```shell
$ kubectl apply -f traefik.yaml
$ kubectl get ingressroute -n monitoring
```

</details>

삽질을 통해 결국 해냈다!
우선 [이 글](https://zerohertz.github.io/k8s-ingress/)에서 확인할 수 있듯 Traefik으로 HTTPS protocol을 하는 법을 익혔고 거기에 path를 추가하여 성공했다!
Path를 설정하고 HTTPS로 monitoring service를 배포하는 방법은 아래와 같다.

우선 path를 Grafana service 내부에서도 인지해야하기 때문에 아래와 같은 설정 파일을 만들고 `ConfigMap`을 생성한다.

```ini grafana.ini
[server]
root_url = https://${DDNS}/grafana
```

```shell
$ kubectl create configmap grafana --from-file=grafana.ini=configmap/grafana.ini -n monitoring
```

위에서 선언한 `ConfigMap`을 `Deployment`에 알려주고, path를 `Middleware`와 `IngressRoute`에 설정하면 끝이다.

```yaml grafana.yaml
...
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  ...
  template:
    ...
    spec:
      containers:
          ...
          volumeMounts:
          ...
          - name: grafana
            mountPath: /etc/grafana/grafana.ini
            subPath: grafana.ini
      volumes:
      ...
      - name: grafana
        configMap:
          name: grafana
---
...
---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: grafana
  namespace: monitoring
spec:
  stripPrefix:
    prefixes:
      - "/grafana"
---
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: grafana
  namespace: monitoring
spec:
  entryPoints:
    - websecure
  routes:
  - match: Host(`${DDNS}`) && PathPrefix(`/grafana`)
    kind: Rule
    middlewares:
    - name: grafana
    services:
    - name: grafana
      port: 80
  tls:
    certResolver: myresolver
```

![HTTPS-grafana](/images/home-server-monitoring/259379473-47a797e9-e546-4bc6-ae21-5e70f17fdb6c.png)