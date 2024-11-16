---
title: K9s
date: 2023-08-08 23:14:04
categories:
- 3. DevOps
tags:
- Kubernetes
- Home Server
---
# Introduction

Kubernetes는 container화된 application의 deployment, scaling 및 orchestration을 단순화하지만 관리 및 troubleshooting이 쉽지 않기 때문에 [K9s](https://k9scli.io/)를 사용한다!
K9s는 terminal 기반의 Kubernetes 관리 도구이며 CLI에서 전체 Kubernetes cluster의 resource를 시각적으로 탐색 및 관리할 수 있다.
주요 기능은 아래와 같다.

1. Real-time Monitoring
   + 주요 resource를 실시간으로 모니터링
   + CPU, memory 사용량 등
2. Resource Management
   + Pod, service, deployment 등의 resource에 대해 조회, 생성, 수정, 삭제 가능
   + YAML 파일을 직접 편집 혹은 명령 실행을 통해 관리 가능
3. Log Exploration
   + 특정 pod 혹은 container의 log를 실시간으로 조회하고 필터링
4. Troubleshooting
   + Kubernetes의 cluster에 발생한 문제를 신속하게 파악하고 해결할 수 있는 다양한 도구 존재

<!-- More -->

---

# Installation

설치는 아주 간단하고 쉽다.
[여기](https://github.com/derailed/k9s/releases)에서 배포 버전을 확인하고 아래의 shell script를 실행하여 원하는 버전을 입력하면 된다.

```bash install_k9s.sh
echo "Insert Version (v0.'00.0'):"
read ver
echo "Download Start!"
echo https://github.com/derailed/k9s/releases/download/v0.${ver}/k9s_Linux_amd64.tar.gz

wget https://github.com/derailed/k9s/releases/download/v0.${ver}/k9s_Linux_amd64.tar.gz
tar -zxvf ./k9s_Linux_amd64.tar.gz
mkdir -p ~/.local/bin
mv ./k9s ~/.local/bin && chmod +x ~/.local/bin/k9s
rm ./k9s_Linux_amd64.tar.gz LICENSE README.md
```

위 script의 실행이 끝나면 `~/.bashrc` 혹은 `~/.zshrc`에 아래 환경 변수를 추가하면 끝이다!

```bash ~/.zshrc
export PATH=$PATH:$HOME/.local/bin
```

```shell
$ source ~/.zshrc
```

---

# Hands-on

실행도 매우 간단하다!

```shell
$ k9s
```

해당 명령어를 입력하면 `K9sCLI`라는 화려한 CLI 이후에 현재 실행되고 있는 pod들이 나란히 명시되어 있다.

![k9s](/images/k9s/k9s.png)

단순히 원하는 pod를 방향키와 enter로 지정한 뒤 아래와 같이 실시간 로그를 확인할 수 있다!

![k9s](/images/k9s/k9s.gif)

~~K9s를 사용하며 느낀점: 아는 만큼 보인다 (아무 것도 안보임)~~