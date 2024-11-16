---
title: Docker Builds in Jenkins using Kaniko for CI/CD Pipelines
date: 2023-11-30 23:18:22
categories:
- 3. DevOps
tags:
- GitHub
- GitHub Actions
- Docker
- Kubernetes
- Jenkins
- CI/CD
- Home Server
---
# Introduction

Jenkins를 통해 GitOps 기반으로 Docker image가 빌드되고 Docker Hub에 push 되게 하려면 아래와 같은 여러 방식이 존재한다.

|                    |            Docker-in-Docker (DinD)             |              Docker Outside of Docker (DooD)              |                            Kaniko                            |
| :----------------: | :--------------------------------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: |
|     Definition     |   Container 내부에 별도의 Docker daemon 실행   |                Host의 Docker daemon을 사용                |           Docker daemon 없이 container image 빌드            |
|      Security      | 더 높은 격리 제공, 하지만 보안상의 우려도 존재 | Host Docker와 직접적인 상호작용으로 보안상 취약할 수 있음 |            Docker daemon 없이 작동하여 보안 강화             |
|    Performance     |              성능 overhead 가능성              |                  일반적으로 더 나은 성능                  | Docker daemon을 사용하지 않기 때문에 성능이 최적화될 수 있음 |
|     Complexity     |             설정과 관리가 더 복잡              |                  상대적으로 간단한 설정                   |      환경 설정에 따라 다르나, 일반적으로 설정이 간단함       |
|     Used Tools     |                  Jib, Buildah                  |                Docker CLI, Docker Compose                 |               Kaniko CLI, Kubernetes와의 통합                |
| Suitable Use Cases |   격리된 환경에서의 독립적인 container 관리    |                간단한 CI/CD pipeline 구성                 |          Cloud 환경 및 Kubernetes에서의 image build          |

<!-- More -->

---

# Kaniko Setup

```shell
$ echo -n '${DOCKER_HUB_USER}:${DOCKER_HUB_TOKEN}' | base64
```

여기서 출력된 결과 (`${DOCKER_HUB_BASE64}`)를 아래와 같이 JSON file로 작성한다.

```json config.json
{
  "auths": {
    "https://index.docker.io/v1/": {
      "auth": "${DOCKER_HUB_BASE64}"
    }
  }
}
```

아래와 같이 해당 JSON file을 다시 base64로 encoding 한다.

```shell
$ cat config.json | base64
```

여기서 출력된 결과 (`${CONFIG_JSON_BASE64}`)를 아래와 같이 Kubernetes secret으로 적용한다.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: docker-config
  namespace: jenkins
type: Opaque
data:
  config.json: ${CONFIG_JSON_BASE64}
```

Test를 위해 아래와 같은 Jenkinsfile을 작성했고 Docker image build 및 Docker Hub로 잘 push 되는 것을 확인했다.

```groovy Jenkinsfile
pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins/agent-type: kaniko
spec:
  containers:
    - name: jnlp
      image: jenkins/inbound-agent:latest
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: kaniko
      image: gcr.io/kaniko-project/executor:debug
      command:
        - /busybox/cat
      tty: true
      resources:
        requests:
          memory: "2048Mi"
          cpu: "2000m"
        limits:
          memory: "4096Mi"
          cpu: "4000m"
      volumeMounts:
        - name: docker-config
          mountPath: /kaniko/.docker/
  volumes:
    - name: docker-config
      secret:
        secretName: docker-config
            """
        }
    }

    environment {
        DOCKERHUB_USERNAME = "zerohertzkr"
        IMAGE_NAME = "test"
    }

    stages {
        stage("Build Docker Image & Push to Docker Hub") {
            steps {
                container("kaniko") {
                    script {
                        def context = "."
                        def dockerfile = "Dockerfile"
                        def image = "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest"

                        sh "/kaniko/executor --context ${context} --dockerfile ${dockerfile} --destination ${image}"
                    }
                }
            }
        }
    }

    post {
        always {
            echo "The process is completed."
        }
    }
}
```

---

# From [GitHub Actions](https://github.com/Zerohertz/docker/tree/c94feeeff9a3e2ac4d85a9115af7282c06487cc1) to [Jenkins](https://github.com/Zerohertz/docker)

기존에는 아래와 같이 GitHub Actions를 통해 GitOps 기반의 Docker CI/CD를 수행했다.

<details>
<summary>
GitHub Actions
</summary>

```yaml .github/workflows/ci.yaml
name: CI

on:
  push:
    branches:
      - main
    paths:
      - "airflow-*/**"

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - name: Check out code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - name: Detect changed directories
        id: set-matrix
        run: |
          CHANGED_DIRS=()
          for path in airflow-*; do
            if git diff --name-only HEAD^ HEAD | grep -qE "^$path/"; then
              CHANGED_DIRS+=("$path")
            fi
          done
          echo "matrix=$(echo ${CHANGED_DIRS[@]} | jq -Rc 'split(" ")')" >> $GITHUB_OUTPUT

  build:
    needs: detect-changes
    if: needs.detect-changes.outputs.matrix != '[]'
    name: Build and Push Docker Image for ${{ matrix.directory }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        directory: ${{fromJson(needs.detect-changes.outputs.matrix)}}
    steps:
      - name: Check out code again
        uses: actions/checkout@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v3
        with:
          context: ${{ matrix.directory }}
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/${{ matrix.directory }}:latest

      - name: Notify Slack
        run: |
          curl -X POST https://slack.com/api/chat.postMessage \
          -H "Authorization: Bearer ${{ secrets.SLACK_BOT_TOKEN }}" \
          -H "Content-type: application/json" \
          -d '{
                  "channel": "zerohertz",
                  "text": ":tada: [GitHub Actions] Build <https://hub.docker.com/repository/docker/${{ secrets.DOCKER_USERNAME }}/${{ matrix.directory }}/general|${{ secrets.DOCKER_USERNAME }}/${{ matrix.directory }}:latest> Completed.",
                  "username": "GitHub",
                  "icon_url": "https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286302856-85c52548-7217-4412-a5cb-a066f588fc13.png",
              }'
```

</details>


특정 이름으로 시작하는 directory (`airflow-*`)에 변경이 존재하면 build 후 Docker Hub에 push하는 workflow다.
이를 Jenkinsfile로 구현하면 아래와 같다.

<details>
<summary>
Jenkinsfile
</summary>

```groovy Jenkinsfile
void setBuildStatus(String message, String state, String context) {
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/Zerohertz/docker"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: context],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins/agent-type: kaniko
spec:
  containers:
    - name: jnlp
      image: jenkins/inbound-agent:latest
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: ubuntu
      image: ubuntu:latest
      command:
        - sleep
      args:
        - "infinity"
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: kaniko
      image: gcr.io/kaniko-project/executor:debug
      command:
        - /busybox/cat
      tty: true
      resources:
        requests:
          memory: "2048Mi"
          cpu: "2000m"
        limits:
          memory: "4096Mi"
          cpu: "4000m"
      volumeMounts:
        - name: docker-config
          mountPath: /kaniko/.docker/
  volumes:
    - name: docker-config
      secret:
        secretName: docker-config
            """
        }
    }
    environment {
        DOCKERHUB_USERNAME = "zerohertzkr"
        CHANGE_PATTERNS = "airflow-*" // ${DIR_NAME}-*,${DIR_NAME}-*,...
        DEFAULT_TAG = "v1.0.0"
    }
    stages {
        stage("Detect Changes") {
            steps {
                script {
                    try {
                        setBuildStatus("Detect...", "PENDING", "$STAGE_NAME")
                        def patterns = env.CHANGE_PATTERNS.split(",")
                        def regex = patterns.collect { it.trim() }.join("|")
                        changedDirs = sh(script: "git diff --name-only HEAD^ HEAD | grep -E '${regex}' | xargs -r -n 1 dirname | uniq", returnStdout: true).trim().split("\n")
                        slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nChange Dirs: ${changedDirs}")
                        setBuildStatus("Success", "SUCCESS", "$STAGE_NAME")
                    } catch (Exception e) {
                        def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                        setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                        slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nError Message: ${STAGE_ERROR_MESSAGE}")
                        throw e
                    }
                }
            }
        }
        stage("Kaniko") {
            when {
                expression {
                    return changedDirs != [""]
                }
            }
            steps {
                script {
                    container("ubuntu") {
                        sh """
                            apt-get update
                            apt-get install -y curl jq
                        """
                    }
                    for (dir in changedDirs) {
                        def imageName = dir.replaceAll("/", "-")
                        def newTag = ""
                        container("ubuntu") {
                            def apiResponse = sh(script: """
                                curl -s "https://hub.docker.com/v2/repositories/${DOCKERHUB_USERNAME}/${imageName}/tags/?page_size=100"
                            """, returnStdout: true).trim()
                            echo "Docker Hub API Response: ${apiResponse}"
                            if (apiResponse.contains("httperror 404")) {
                                newTag = env.DEFAULT_TAG
                            } else {
                                def currentTag = sh(script: "echo '${apiResponse}' | jq -r '.results[].name' | sort -V | grep v | tail -n 1", returnStdout: true).trim()
                                echo "Current Tag: ${currentTag}"
                                def version = currentTag.replaceAll("[^0-9.]", "")
                                def (major, minor, patch) = version.tokenize(".").collect { it.toInteger() }
                                newTag = "v${major}.${minor}.${patch + 1}"
                            }
                            echo "New Tag: ${newTag}"
                        }
                        container("kaniko") {
                            script {
                                try {
                                    setBuildStatus("Build...", "PENDING", "$STAGE_NAME - ${DOCKERHUB_USERNAME}/${imageName}:${newTag}")
                                    sh "/kaniko/executor --context ${dir} --dockerfile ${dir}/Dockerfile --destination ${DOCKERHUB_USERNAME}/${imageName}:latest --cleanup && mkdir -p /workspace"
                                    sh "/kaniko/executor --context ${dir} --dockerfile ${dir}/Dockerfile --destination ${DOCKERHUB_USERNAME}/${imageName}:${newTag} --cleanup && mkdir -p /workspace"
                                    setBuildStatus("Success", "SUCCESS", "$STAGE_NAME - ${DOCKERHUB_USERNAME}/${imageName}:${newTag}")
                                    slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nIMAGE: <https://hub.docker.com/repository/docker/zerohertzkr/${imageName}/general|${DOCKERHUB_USERNAME}/${imageName}:${newTag}>")
                                } catch (Exception e) {
                                    def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                                    setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME - ${DOCKERHUB_USERNAME}/${imageName}:${newTag}")
                                    slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> FAIL\nBRANCH NAME: ${env.BRANCH_NAME}\nIMAGE: ${DOCKERHUB_USERNAME}/${imageName}:${newTag}\nError Message: ${STAGE_ERROR_MESSAGE}")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

</details>

![github-slack](/images/jenkins-kaniko/github-slack.png)

1. Detect Changes
   + `git diff --name-only HEAD^ HEAD | grep -E '${regex}' | xargs -r -n 1 dirname | uniq"`으로 변경 사항이 존재하는 목표 directory를 불러온다.
2. Kaniko
   + `curl -s "https://hub.docker.com/v2/repositories/${DOCKERHUB_USERNAME}/${imageName}/tags/?page_size=100"`를 통해 현재 Docker Hub에 존재하는 version의 이름을 불러온다.
     + Tag가 `v${major}.${minor}.${patch}`의 format을 따르지 않거나 Docker Hub에 존재하지 않는다면 push할 tag를 `v1.0.0`으로 설정한다.
     + Tag가 `v${major}.${minor}.${patch}`의 format을 따르면 `v${major}.${minor}.${patch + 1}`으로 push한다.
   + `sh "/kaniko/executor --context ${dir} --dockerfile ${dir}/Dockerfile --destination ${DOCKERHUB_USERNAME}/${imageName}:${newTag} --cleanup && mkdir -p /workspace"`으로 다음 version의 image를 push한다.
   + `sh "/kaniko/executor --context ${dir} --dockerfile ${dir}/Dockerfile --destination ${DOCKERHUB_USERNAME}/${imageName}:latest --cleanup && mkdir -p /workspace"`으로 `latest` tag를 사용할 수 있게 동일한 image를 push한다.

---

# Errors

## Error: unknown command "/kaniko/executor --context . --dockerfile Dockerfile --no-push" for "executor"

<details>
<summary>
Jenkinsfile
</summary>

```groovy Jenkinsfile
pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins/agent-type: kaniko
spec:
  containers:
    - name: jnlp
      image: jenkins/inbound-agent:latest
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: kaniko
      image: gcr.io/kaniko-project/executor:latest
      args: ["--context", ".", "--dockerfile", "Dockerfile", "--no-push"]
      resources:
        requests:
          memory: "2048Mi"
          cpu: "2000m"
        limits:
          memory: "4096Mi"
          cpu: "4000m"
      volumeMounts:
        - name: docker-config
          mountPath: /kaniko/.docker/
  volumes:
    - name: docker-config
      secret:
        secretName: docker-config
            """
        }
    }
...
```

</details>


위와 같이 Jenkinsfile을 설정하면 항상 오류가 발생한다.
그 이유는 agent pod를 생성할 때 kaniko container의 `args`에 `--dockerfile`로 경로를 지정해야하는데 multibranch pipeline은 agent 생성 후 ~ stage 시작 전에 원격 저장소에서 Dockerfile을 포함한 code들을 불러오기 때문에 아래와 같은 오류가 발생한다.

```bash
Error: unknown command "/kaniko/executor --context . --dockerfile Dockerfile --no-push" for "executor"
```

## Cleanup

아래 Jenkinsfile과 같이 Kaniko를 이용해 한 개의 Dockerfile을 build 하고 사용하면 잘 작동하지만, 여러 Dockerfile들을 build 하고 사용해보면 오류가 발생한다.

<details>
<summary>
Jenkinsfile
</summary>

```groovy Jenkinsfile
void setBuildStatus(String message, String state, String context) {
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/Zerohertz/docker"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: context],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins/agent-type: kaniko
spec:
  containers:
    - name: jnlp
      image: jenkins/inbound-agent:latest
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: ubuntu
      image: ubuntu:latest
      command:
        - sleep
      args:
        - "infinity"
      resources:
        requests:
          memory: "512Mi"
          cpu: "500m"
        limits:
          memory: "1024Mi"
          cpu: "1000m"
    - name: kaniko
      image: gcr.io/kaniko-project/executor:debug
      command:
        - /busybox/cat
      tty: true
      resources:
        requests:
          memory: "2048Mi"
          cpu: "2000m"
        limits:
          memory: "4096Mi"
          cpu: "4000m"
      volumeMounts:
        - name: docker-config
          mountPath: /kaniko/.docker/
  volumes:
    - name: docker-config
      secret:
        secretName: docker-config
            """
        }
    }
    environment {
        DOCKERHUB_USERNAME = "zerohertzkr"
        CHANGE_PATTERNS = "airflow-*" // ${DIR_NAME}-*,${DIR_NAME}-*,...
        DEFAULT_TAG = "v1.0.0"
    }
    stages {
        stage("Detect Changes") {
            steps {
                script {
                    try {
                        setBuildStatus("Detact...", "PENDING", "$STAGE_NAME")
                        def patterns = env.CHANGE_PATTERNS.split(",")
                        def regex = patterns.collect { it.trim() }.join("|")
                        changedDirs = sh(script: "git diff --name-only HEAD^ HEAD | grep -E '${regex}' | xargs -r -n 1 dirname | uniq", returnStdout: true).trim().split("\n")
                        slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nChange Dirs: ${changedDirs}")
                        setBuildStatus("Success", "SUCCESS", "$STAGE_NAME")
                    } catch (Exception e) {
                        def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                        setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME")
                        slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nError Message: ${STAGE_ERROR_MESSAGE}")
                        throw e
                    }
                }
            }
        }
        stage("Kaniko") {
            when {
                expression {
                    return changedDirs != [""]
                }
            }
            steps {
                script {
                    container("ubuntu") {
                        sh """
                            apt-get update
                            apt-get install -y curl jq
                        """
                    }
                    for (dir in changedDirs) {
                        def imageName = dir.replaceAll("/", "-")
                        def newTag = ""
                        container("ubuntu") {
                            def apiResponse = sh(script: """
                                curl -s "https://hub.docker.com/v2/repositories/${DOCKERHUB_USERNAME}/${imageName}/tags/?page_size=100"
                            """, returnStdout: true).trim()
                            if (apiResponse.contains("httperror 404")) {
                                newTag = env.DEFAULT_TAG
                            } else {
                                def currentTag = sh(script: "echo '${apiResponse}' | jq -r '.results[].name' | sort -V | tail -n 1", returnStdout: true).trim()
                                try {
                                    def (major, minor, patch) = currentTag.tokenize(".").collect { it.toInteger() }
                                    newTag = "v${major}.${minor}.${patch + 1}"
                                } catch (Exception e) {
                                    newTag = env.DEFAULT_TAG
                                }
                            }
                        }
                        container("kaniko") {
                            script {
                                def dockerfile = "Dockerfile"
                                def image = "${DOCKERHUB_USERNAME}/${imageName}:${newTag}"
                                try {
                                    setBuildStatus("Build...", "PENDING", "$STAGE_NAME: $image")
                                    sh "/kaniko/executor --context ${dir} --dockerfile ${dir}/${dockerfile} --destination ${image}"
                                    setBuildStatus("Success", "SUCCESS", "$STAGE_NAME: $image")
                                    slackSend(color: "good", message: ":+1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nIMAGE: <https://hub.docker.com/repository/docker/zerohertzkr/${imageName}/general|${image}>")
                                } catch (Exception e) {
                                    def STAGE_ERROR_MESSAGE = e.getMessage().split("\n")[0]
                                    setBuildStatus(STAGE_ERROR_MESSAGE, "FAILURE", "$STAGE_NAME: $image")
                                    slackSend(color: "danger", message: ":-1:  <${env.BUILD_URL}|[${env.JOB_NAME}: ${STAGE_NAME}]> SUCCESS\nBRANCH NAME: ${env.BRANCH_NAME}\nIMAGE: ${image}\nError Message: ${STAGE_ERROR_MESSAGE}")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

</details>

Build에 실패하면 아래와 같은 log들이 출력되며 성공하더라도 두 번째 이후로 build된 image는 `RUN pip install -r requirements.txt`와 같은 명령어가 제대로 적용되지 않아 정상적으로 사용할 수 없다.

<details>
<summary>
Jenkins error logs
</summary>

```bash
Traceback (most recent call last):
  File "/usr/bin/py3clean", line 210, in <module>
    main()
  File "/usr/bin/py3clean", line 196, in main
    pfiles = set(dpf.from_package(options.package))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/share/python3/debpython/files.py", line 53, in from_package
    raise Exception("cannot get content of %s" % package_name)
Exception: cannot get content of libglib2.0-dev-bin
error running python rtupdate hook libglib2.0-dev-bin
dpkg: error processing package python3 (--configure):
 installed python3 package post-installation script subprocess returned error exit status 4
dpkg: dependency problems prevent configuration of python3-wheel:
 python3-wheel depends on python3:any; however:
  Package python3 is not configured yet.

dpkg: error processing package python3-wheel (--configure):
 dependency problems - leaving unconfigured
Setting up perl (5.36.0-7) ...
Setting up libgprofng0:amd64 (2.40-2) ...
dpkg: dependency problems prevent configuration of python3-dev:
 python3-dev depends on python3 (= 3.11.2-1+b1); however:
  Package python3 is not configured yet.

dpkg: error processing package python3-dev (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of python3-gi:
 python3-gi depends on python3 (<< 3.12); however:
  Package python3 is not configured yet.
 python3-gi depends on python3 (>= 3.11~); however:
  Package python3 is not configured yet.
 python3-gi depends on python3:any; however:
  Package python3 is not configured yet.

dpkg: error processing package python3-gi (--configure):
 dependency problems - leaving unconfigured
Setting up libgcc-12-dev:amd64 (12.3.0-1ubuntu1~22.04) ...
dpkg: dependency problems prevent configuration of python3-pip:
 python3-pip depends on python3-wheel; however:
  Package python3-wheel is not configured yet.
 python3-pip depends on python3:any; however:
  Package python3 is not configured yet.

dpkg: error processing package python3-pip (--configure):
 dependency problems - leaving unconfigured
Setting up libjs-sphinxdoc (5.3.0-4) ...
Setting up libdpkg-perl (1.21.22) ...
Setting up libx265-199:amd64 (3.5-2+b1) ...
Setting up libhtml-parser-perl:amd64 (3.81-1) ...
Setting up libc6-dev:amd64 (2.36-9+deb12u3) ...
dpkg: dependency problems prevent configuration of python3-lib2to3:
 python3-lib2to3 depends on python3:any (>= 3.10.8-0~); however:
  Package python3 is not configured yet.
 python3-lib2to3 depends on python3:any (<< 3.12); however:
  Package python3 is not configured yet.

dpkg: error processing package python3-lib2to3 (--configure):
 dependency problems - leaving unconfigured
Setting up binutils-x86-64-linux-gnu (2.40-2) ...
Setting up libnet-ssleay-perl:amd64 (1.92-2+b1) ...
dpkg: dependency problems prevent configuration of python3-pkg-resources:
 python3-pkg-resources depends on python3:any; however:
  Package python3 is not configured yet.

dpkg: error processing package python3-pkg-resources (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of python3-distutils:
 python3-distutils depends on python3:any (>= 3.10.8-0~); however:
  Package python3 is not configured yet.
 python3-distutils depends on python3:any (<< 3.12); however:
  Package python3 is not configured yet.
 python3-distutils depends on python3-lib2to3 (= 3.11.2-3); however:
  Package python3-lib2to3 is not configured yet.

dpkg: error processing package python3-distutils (--configure):
 dependency problems - leaving unconfigured
Setting up libxml-parser-perl (2.46-4) ...
dpkg: dependency problems prevent configuration of python3-dbus:
 python3-dbus depends on python3 (<< 3.12); however:
  Package python3 is not configured yet.
 python3-dbus depends on python3 (>= 3.11~); however:
  Package python3 is not configured yet.
 python3-dbus depends on python3:any; however:
  Package python3 is not configured yet.

dpkg: error processing package python3-dbus (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of python3-setuptools:
 python3-setuptools depends on python3-pkg-resources (= 66.1.1-1); however:
  Package python3-pkg-resources is not configured yet.
 python3-setuptools depends on python3-distutils; however:
  Package python3-distutils is not configured yet.
 python3-setuptools depends on python3:any; however:
  Package python3 is not configured yet.

dpkg: error processing package python3-setuptools (--configure):
 dependency problems - leaving unconfigured
Setting up libstdc++-12-dev:amd64 (12.3.0-1ubuntu1~22.04) ...
Setting up libfile-fcntllock-perl (0.22-4+b1) ...
Setting up libclone-perl:amd64 (0.46-1) ...
Setting up libalgorithm-diff-perl (1.201-1) ...
Setting up libheif1:amd64 (1.15.1-1) ...
Setting up libnet-dbus-perl (1.2.0-2) ...
Setting up binutils (2.40-2) ...
Setting up dpkg-dev (1.21.22) ...
Setting up libexpat1-dev:amd64 (2.5.0-1) ...
Setting up gcc-12 (12.3.0-1ubuntu1~22.04) ...
Setting up libgd3:amd64 (2.3.3-9) ...
Setting up zlib1g-dev:amd64 (1:1.2.13.dfsg-1) ...
Setting up libalgorithm-diff-xs-perl:amd64 (0.04-8+b1) ...
Setting up libc-devtools (2.36-9+deb12u3) ...
Setting up libalgorithm-merge-perl (0.08-5) ...
Setting up g++-12 (12.3.0-1ubuntu1~22.04) ...
Setting up gcc (4:12.2.0-3) ...
Setting up libpython3.11-dev:amd64 (3.11.2-6) ...
Setting up g++ (4:12.2.0-3) ...
Setting up build-essential (12.9ubuntu3) ...
Setting up libpython3-dev:amd64 (3.11.2-1+b1) ...
Setting up python3.11-dev (3.11.2-6) ...
Processing triggers for libc-bin (2.36-9+deb12u3) ...
Errors were encountered while processing:
 python3
 python3-wheel
 python3-dev
 python3-gi
 python3-pip
 python3-lib2to3
 python3-pkg-resources
 python3-distutils
 python3-dbus
 python3-setuptools
E: Sub-process /usr/bin/dpkg returned an error code (1)
error building image: error building stage: failed to execute command: waiting for process to exit: exit status 100
```

</details>

여러 image를 build하는 과정에서 layer 단위의 정보를 다시 가져와서 생기는 오류로 파악된다.
따라서 [이와 같이](https://github.com/GoogleContainerTools/kaniko/issues/1586#issuecomment-1835450877) `--cleanup && mkdir -p /workspace`를 추가해야한다.
