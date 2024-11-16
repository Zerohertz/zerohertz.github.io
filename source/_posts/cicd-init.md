---
title: GitHub Actions와 Argo CD 기반 CI/CD 도전기
date: 2023-08-23 08:42:58
categories:
- 3. DevOps
tags:
- CI/CD
- GitHub
- GitHub Actions
- Argo CD
- Kubernetes
- Home Server
---
# Introduction

+ CI/CD
  + CI (Continuous Integration)
    + 개발자들이 자주 (일반적으로 하루에 여러 번) 자신의 코드 변경 사항을 메인 브랜치에 통합하는 프로세스
    + 이는 코드의 통합 문제를 조기에 견하고 해결하는 데 도움이 됩니다.
  + CD (Continuous Deployment/Delivery)
    + 자동화된 테스트를 통과한 모든 코드 변경 사항이 자동으로 프로덕션 환경에 배포되는 프로세스 (Continuous Deployment) 또는 프로덕션 환경에 배포될 준비가 되면 수동으로 배포될 수 있게 하는 프로세스 (Continuous Delivery).
+ GitHub Actions: GitHub에서 제공하는 CI/CD 플랫폼
  + Workflow를 `.yml` 또는 `.yaml` 파일로 정의하여 코드의 빌드, 테스트, 배포 등 다양한 자동화 작업 수행
+ Argo CD: Kubernetes를 위한 선언적, GitOps 연속 배포 도구
  + Git 저장소에 정의된 Kubernetes 명세를 cluster와 동기화하는 데 사용
  + Application의 배포 상태, 구성 차이 및 동기화 상태 시각화 대시보드 제공

이러한 GitHub Actions와 Argo CD를 통해 아래와 같은 CI/CD를 도전해본다.

![cicd](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262577036-f2ad5f3b-6f45-4f52-8368-20e3fa97f22c.png)

<!-- More -->

---

# GitHub Actions

## Hands-on

GitHub Actions를 통한 CI를 테스트하기 위해 아래 파일들을 준비했다.

```bash
.
├── Dockerfile
├── .github
│   └── workflows
│       └── ci.yaml
└── main.py
```

```docker Dockerfile
FROM python:3.8

WORKDIR /app
COPY main.py .

CMD ["python", "main.py"]
```

```python main.py
print("Hello, World!")
```

GitHub Actions는 `.github/workflows/*.yaml`에서 어떤 작업들을 언제 진행할지 정의할 수 있다.
위에서 정의한 `Dockerfile`을 build하고 push하기 위해 [Docker Hub](https://hub.docker.com/)에 login한다.
이를 위해 아래와 같이 GitHub Actions에서 Docker Hub에 login하기 위해 사용할 변수들을 정의한다.

![secrets](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262512736-5a622851-1e6d-41e4-8241-d53a24ae4d31.png)

```yaml .github/workflows/ci.yaml
name: Build and Push Docker Image

on:
  push:
    branches:
      - master

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Check out code
      uses: actions/checkout@v3

    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: zerohertzkr/build-test:latest
```

이 파일을 `master` branch에 push하면 build 후 Docker Hub에 잘 push되는 것을 확인할 수 있다.

![build-test:latest](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262511256-f2c4a095-69da-40c2-93ef-1ef5f8942925.png)

```shell
$ docker run --name build-test zerohertzkr/build-test
...
Status: Downloaded newer image for zerohertzkr/build-test:latest
Hello, World!
```

아래와 같이 변경 후 `master` branch에 push하고 다시 `docker run`을 하게되면 아래와 같이 새로 build된 image임을 확인할 수 있다.

```python main.py
print("Hello, World! => CI!")
```

```shell
$ git push origin master
...
$ docker run --name build-test zerohertzkr/build-test
...
Status: Downloaded newer image for zerohertzkr/build-test:latest
Hello, World! => CI!
```

## Portal

![server-portal](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262010647-c873b57a-95cb-4a57-ac44-6116b8d3c265.png)

[Home server portal](https://zerohertz.xyz)의 [repository](https://github.com/Zerohertz/server-portal)에 push하면 Docker Hub에 build된 이미지를 push하도록 설정했다.
또한 Argo CD에 Docker image 버전 변경을 알리기 위해 `k8s/main.yaml`에서 사용하는 버전을 바꾸고 push하도록 설정했다.

```yaml .github/workflows/ci.yaml
name: CI

on:
  push:
    branches:
      - main

jobs:
  build:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.message, 'Build')
    outputs:
      TAG: ${{ steps.extract_tag.outputs.TAG }}
    steps:
      - name: Check out code
        uses: actions/checkout@v3

      - name: Extract tag from commit message
        id: extract_tag
        run: |
          TAG=$(echo "${{ github.event.head_commit.message }}" | grep -oP 'Build: \s*\K[\w\.]+')
          echo "Extracted tag is $TAG"
          echo "TAG=$TAG" >> $GITHUB_ENV
          echo "TAG=$TAG" >> $GITHUB_OUTPUT

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v3
        with:
          context: .
          push: true
          tags: zerohertzkr/server-portal:${{ env.TAG }}

  push:
    name: Push Manifest
    needs: build
    runs-on: ubuntu-latest
    env:
      TAG: ${{ needs.build.outputs.TAG }}
    steps:
      - name: Check out code
        uses: actions/checkout@v3
        with:
          token: ${{ secrets.GH_TOKEN }}

      - name: Change manifest
        run: sed -i "s|zerohertzkr/server-portal:[^ ]*|zerohertzkr/server-portal:${{ env.TAG }}|" k8s/main.yaml

      - name: git push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git config credential.helper store
          git add k8s/main.yaml
          git commit -m ":tada: Update: Image [${{ env.TAG }}]"
          git push
```

---

# Argo CD

Home server portal를 Argo CD로 배포하기 위해 아래와 같이 설정했다.

![argo-cd-1](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262567537-e280e68c-2f65-4a2d-b3e5-58c4e062c2ea.png)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: portal
spec:
  destination:
    name: ''
    namespace: portal
    server: 'https://kubernetes.default.svc'
  source:
    path: ./k8s
    repoURL: 'https://github.com/Zerohertz/server-portal'
    targetRevision: HEAD
  sources: []
  project: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

그리고 테스트를 위해 아래 사항들을 변경 후 배포해봤다.

```ts src/common/constants/app.ts
import type { AppInterface } from "@/common/types/app";

export const APP_LIST: AppInterface[] = [
  ...
  {
    name: "TEST",
    href: "https://zerohertz.xyz",
    imageSrc: "/favicon.png",
  },
];
```

```shell
$ git add src/common/constants/app.ts
$ git commit -m ":memo: Build: test"
$ git push origin main
```

아주 잘 배포되는 것을 확인할 수 있다.

![result](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262576978-2b89e1bf-8f06-4b05-91cf-8bdac9bd582c.png)

![github-actions](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262684482-ac3f0e8a-a426-4b4d-be76-04783395fc2c.gif)
![argo-cd-2](https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/262684467-c1ece4a8-0b6e-4ac7-909b-4acd22878eec.gif)