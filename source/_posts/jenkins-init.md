---
title: Jenkins Installation and Initial Setup on Kubernetes
date: 2023-11-01 19:39:01
categories:
- 3. DevOps
tags:
- Kubernetes
- Jenkins
- CI/CD
- Home Server
---
# Installation (Helm)

Argo CD로 Jenkins를 배포하기 위해 Helm으로 아래와 같이 helm chart를 받는다.

```
$ helm repo add jenkins https://charts.jenkins.io
$ helm repo update
$ helm pull jenkins/jenkins
$ tar -xvf jenkins-4.8.2.tgz
```

Traefik을 통해 ingress 설정을 `https://jenkins.zerohertz.xyz`로 정의했다.
따라서 `values.yaml` 파일을 아래와 같이 수정하고 배포했다.

```yaml values.yaml
...
controller:
  ...
  jenkinsUrlProtocol: "https"
  jenkinsUrl: "jenkins.zerohertz.xyz"
  ...
  JCasC:
    ...
    securityRealm: |-
      local:
        allowsSignup: false
        enableCaptcha: false
        users:
        - id: "${chart-admin-username}"
          name: "Jenkins Admin"
          password: "${chart-admin-password}"
...
```

![tada](/images/jenkins-init/tada.png)

<!-- More -->

---

# Plugins

Jenkins에는 유용한 plugin들이 존재한다.
Plugin은 아래와 같이 페이지를 이동하여 조회, 설치, 업데이트를 진행할 수 있다.

![plugins](/images/jenkins-init/plugins.png)

## [GitHub](https://plugins.jenkins.io/github/)

효율적인 CI/CD를 위해 GitOps를 사용하려면 GitHub와의 연동이 필요하다.
따라서 해당 plugin을 다운로드하여 아래와 같이 설정했다.

![github](/images/jenkins-init/github.png)

## [Google Login](https://plugins.jenkins.io/google-login/)

Helm으로 배포하고 그대로 사용하면 로그인 제한이 없기 때문에 보안이 취약하다.
따라서 Traefik의 middleware를 사용해서 Google OAuth를 사용했지만 이렇게 사용하면 GitHub Hooks가 Jenkins에 접근을 하지못하기 때문에 Jenkins에서 사용할 수 있는 plugin을 설치했다.

![google-login-1](/images/jenkins-init/google-login-1.png)

[이 글](https://zerohertz.github.io/traefik-oauth/)을 참고하여 아래와 같이 web application을 생성한다.

![google-login-2](/images/jenkins-init/google-login-2.png)

그리고 이 application의 `ID`와 `Secret`을 입력하면 OAuth 화면을 확인할 수 있다.

![google-login-3](/images/jenkins-init/google-login-3.png)

## [Role-based Authorization Strategy](https://plugins.jenkins.io/role-strategy/)

그러나 위의 Google Login plugin은 모든 인증된 사용자가 admin 권한을 가진다.
따라서 이를 적절히 제한하기 위해 plugin을 설치하고 아래와 같이 설정하여 `Manage and Assign Roles`를 활성화시켰다.

![role-auth-1](/images/jenkins-init/role-auth-1.png)

해당 메뉴에 들어가면 그룹을 나눠 role을 정의할 수 있다.
아래는 Google 인증을 받은 사용자는 읽기 권한만, 내 계정은 admin 권한을 부여했다.

![role-auth-2](/images/jenkins-init/role-auth-2.png)

## [ThinBackup](https://plugins.jenkins.io/thinBackup/)

위의 plugin을 설치하고 실험하던 도중,,, Jenkins의 재부팅 시 설정들이 모두 증발하는 현상을 발견했고, 이를 해결하기 위해 plugin을 아래와 같이 설정했다.

![thinbackup-1](/images/jenkins-init/thinbackup-1.png)

Backup 경로를 정의하면 아래와 같이 backup 버튼을 클릭하여 저장되는 것을 확인할 수 있고 이것을 restore 할 수 있다.
Restore 버튼을 누른 뒤 `Reload Configuration from Disk`를 클릭하면 복원된다.

![thinbackup-2](/images/jenkins-init/thinbackup-2.png)

## [GitHub Branch Source](https://plugins.jenkins.io/github-branch-source/)

GitHub에서 Pull Request 시 main 혹은 master branch에 merge 전 CI를 위해 [GitHub Pull Request Builder](https://plugins.jenkins.io/ghprb/)를 설치하려 했으나, 보안 이슈로 해당 plugin을 설치했다.
그리고 해당 plugin을 사용하면 아래와 같이 원하지 않는 이름의 status들이 표시되는데 이를 방지하기 위해 [Disable GitHub Multibranch Status](https://github.com/jenkinsci/disable-github-multibranch-status-plugin)를 설치했다.

<img src="/images/jenkins-init/github-status.png" alt="github-status" width="530" />

## [Blue Ocean](https://plugins.jenkins.io/blueocean/)

Blue Ocean은 Jenkins Pipeline을 기반으로 처음부터 설계되었으며 다음과 같은 핵심 기능을 통해 팀 구성원 모두의 이해를 돕고 혼란을 줄이며 명확성을 증가시킨다고 한다.

- 소프트웨어 파이프라인의 상태를 빠르고 직관적으로 이해할 수 있도록 pipeline의 정교한 시각화 기능
- 사용자를 이끄는 직관적이고 시각적인 과정을 통해 pipeline을 자동화하는 것을 접근하기 쉽게 해주는 편집기
- DevOps 팀의 각 구성원의 역할 기반 요구 사항에 맞게 Jenkins UI 개인화
- 개입이 필요하거나 문제가 발생했을 때 정확한 지점 식별
- GitHub과 Bitbucket에서 다른 사람들과 코드 협업 시 개발자 생산성을 극대화할 수 있는 브랜치 및 풀 리퀘스트에 대한 네이티브 통합

![blue-ocean](/images/jenkins-init/blue-ocean.png)

## [Slack Notification](https://plugins.jenkins.io/slack/)

```groovy Jenkinsfile
slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nCHANGE TARGET: ${env.CHANGE_TARGET}\nCommit Message:  ${commitMessage}")
```

위와 같은 코드로 아래와 같이 쉽게 Slack으로 message를 전달할 수 있다.

![slack-notification](/images/jenkins-init/slack-notification.png)

---

# Examples

대략적인 설정을 완료했으니 간단한 예제들을 수행해본다.

![create](/images/jenkins-init/create.png)

위와 같이 새로운 project를 생성하면 아래와 같이 여러 양식을 선택할 수 있다.

![pipeline-1](/images/jenkins-init/pipeline-1.png)

여기서 `Pipeline`을 선택한다.

![pipeline-2](/images/jenkins-init/pipeline-2.png)

그리고 아래의 코드를 작성한다.

```groovy
pipeline {
    agent any

    stages {
        stage('Hello') {
            steps {
                echo 'Hello, World!'
            }
        }

        stage('Bye') {
            steps {
                echo 'Bye, World...'
            }
        }
    }
}
```

![sucess](/images/jenkins-init/sucess.png)

이렇게 build가 성공하면 해가 쨍쨍한 것을 확인할 수 있다.

![fail](/images/jenkins-init/fail.png)

실패하면 우중충한 것도 확인할 수 있다!