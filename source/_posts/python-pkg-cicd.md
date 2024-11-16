---
title: Constructing a CI/CD Pipeline for Python Packages Using Jenkins
date: 2023-11-03 18:28:05
categories:
- 3. DevOps
tags:
- Python
- CI/CD
- Jenkins
- GitHub
- GitHub Actions
- Home Server
---
# Introduction

자주 사용하는 Python 함수들을 package로 생성하고 CI/CD pipeline을 Jenkins로 구축하여 자동으로 배포될 수 있게 해보자!
배포는 package 내부의 함수 및 객체를 sphinx로 문서화하고 [PyPI](https://pypi.org/) (Python Package Index)에 업로드하여 `pip`로 설치할 수 있게 해볼 것이다.
CI/CD 계획은 아래와 같다.

+ Dev Branch Push
  1. Lint
  2. Build
  3. Test
+ Master Branch PR
  1. Lint
  2. Build
  3. Test
  4. Docs
+ Master Branch Puah
  1. Build
  2. Deploy
     1. PyPI
     2. GitHub

이 사항들을 Jenkins로 개발하고 blue ocean으로 확인해보면 아래와 같이 구성된다.

![jenkinsfile](/images/python-pkg-cicd/280279733-fd755f7d-5133-4104-92e0-592623b31bb1.png)

<!-- More -->

---

# Scenario

## Dev Branch Push

Release를 위해 `add` 함수를 개발하고 이를 `v1.0`으로 배포하기 위해 `${PACKAGE_NAME}/__init__.py`의 `__version__`을 `"v1.0"`으로 변경하여 `dev-v1.0` branch에 commit 및 push 한다.

![dev](/images/python-pkg-cicd/280287548-6ff1606f-0c39-455b-b2dd-1b80a9dea4e7.gif)

그러면 위와 같이 Lint, Build, Test를 진행한다.
이 과정에서 문제가 생길 시 아래와 같이 오류가 발생한다.

![dev-err](/images/python-pkg-cicd/280291571-06c4dfd9-d5c4-45fb-acff-2de86be00167.png)

## Pull Request

`v1.0`을 위한 모든 개발을 마치면 pull request를 `master` branch로 생성하면 아래와 같이 Lint, Build, Test, Docs를 진행한다.

![pr](/images/python-pkg-cicd/280288434-49f89c34-45dd-4a12-b122-5ae0b589c8fb.gif)

그러면 위와 같이 새로운 pull request가 생성되고, 이는 package에 대한 문서를 빌드한 것이다.
최종 merge 전에 해당 pull request도 merge 해야한다.

## Master Branch Push

배포를 위한 모든 준비를 마쳤다면 merge 버튼을 통해 `master` branch에 push 한다.

![master](/images/python-pkg-cicd/280289923-1d94b631-46a5-4a79-8875-ccecb8fc8ec2.gif)

그러면 위와 같이 GitHub에서 page가 빌드되고, 그와 동시에 package를 build 후 GitHub와 PyPI에 배포한다.

<img width="1000" alt="스크린샷 2023-11-03 오후 10 18 53" src="/images/python-pkg-cicd/280289888-4ef4cd00-9cd6-4513-a0c8-aaf8d2fbe225.png">

```shell
$ pip install zerohertzPkg
Collecting zerohertzPkg
  Downloading zerohertzPkg-1.0-py3-none-any.whl (1.6 kB)
Installing collected packages: zerohertzPkg
Successfully installed zerohertzPkg-1.0
```

이렇게 CI/CD pipeline을 완성해봤다.
그렇다면 어떻게 이런 pipeline을 구축할 수 있을까?
(~~지금까지 수많은 삽질을 결과물로 정리했으니 함께 다시 구축해봐요 ^^~~)

---

# CI/CD Pipeline Setup

## GitHub

![github-1](/images/python-pkg-cicd/280292866-25819808-2c0a-4fe1-a01d-0d8293977b81.png)

GitOps를 수행하기 위한 GitHub repository와 webhook을 생성한다.
Payload URL은 `${PROTOCOL}://${JENKINS_URL}/github-webhook/`으로 작성하고 Jenkins가 수집하기 원하는 `Pushes`와 `Pull requests`에 체크한다.
아래와 같이 초기 개발에 필요한 파일 및 코드들을 `master`에 push 한다.

```bash
├── Jenkinsfile
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── sphinx
│   ├── build
│   │   └── doctrees
│   │       ├── environment.pickle
│   │       └── index.doctree
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── conf.py
│       ├── index.rst
│       ├── _static
│       └── _templates
├── test
└── zerohertzLib
    └── __init__.py
```

이제 바로 `master` branch에 push 하지 않을 예정이기 때문에 아래와 같이 세 옵션을 brnach protection rule로 정의한다.

![github-2](/images/python-pkg-cicd/280301007-f80037cd-9544-4094-9279-0420d34cda5e.png)

1. Require a pull request before merging: Pull request를 통해 merge 가능 (바로 push 불가능)
2. Require status checks to pass before merging: Merge 시 지정한 상태에 이상이 없어야 가능하게 설정
   + Require branches to be up to date before merging: 최신 코드로 테스트 되었는지 확인
3. Do not allow bypassing the above settings: 관리자 권한 유저도 branch protection rule 설정

모든 설정을 마치면 아래와 같이 바로 `master` branch에 push 할 수 없다.

```shell
$ git push
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Delta compression using up to 12 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (2/2), 897 bytes | 897.00 KiB/s, done.
Total 2 (delta 1), reused 0 (delta 0)
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
remote: error: GH006: Protected branch update failed for refs/heads/master.
remote: error: Changes must be made through a pull request.
To https://github.com/Zerohertz/zerohertzLib
 ! [remote rejected] master -> master (protected branch hook declined)
error: failed to push some refs to 'https://github.com/Zerohertz/zerohertzLib'
```

![github-3](/images/python-pkg-cicd/280302828-392ee859-0bc9-45e5-a1c0-5e2be2fbf574.png)

마지막으로 Sphinx로 생성된 문서를 배포하기 위해 GitHub pages를 위와 같이 설정한다.

## Jenkins

GitHub의 모든 설정을 마쳤으니 Jenkinsfile을 통해 CI/CD pipeline이 잘 작동할 수 있게 Jenkins를 설정한다.

![jenkins-setup-1](/images/python-pkg-cicd/280266359-d6f8fbda-71e0-48a3-bde4-1cb8573b469c.png)

Multibranch Pipeline으로 project를 생성한다.

![jenkins-setup-2](/images/python-pkg-cicd/280266846-cf1a5cf6-f3be-48fd-b680-4f3d862007d6.png)

적절한 GitHub credentials와 repository HTTPS URL을 기입한다.
마지막으로 지저분한 UI를 방지하기 위해 [Disable GitHub Notifications](https://zerohertz.github.io/jenkins-init/#GitHub-Branch-Source)를 설정했다.
이제 모든 설정은 끝났다!
해당 CI/CD pipeline이 적용된 코드들은 [Zerohertz/zerohertzLib](https://github.com/Zerohertz/zerohertzLib)에서 확인할 수 있다.

---

# Updates

이후에 CI/CD pipeline에 부족한 점이 많아 수정을 진행했다.
Package의 update 시 변경 사항들을 한 눈에 볼 수 있게 [GitHub API로 불러오고 Release Notes를 생성하는 코드](https://zerohertz.github.io/zerohertzLib/zerohertzLib.api.html#zerohertzLib.api.GitHub.release_note)를 추가했다.
최신 CI/CD pipeline의 설명은 [여기](https://zerohertz.github.io/zerohertzLib/cicd.html)에서 확인할 수 있고, 위의 코드로 생성된 release notes는 [여기](https://zerohertz.github.io/zerohertzLib/release.html)에서 확인할 수 있다.

---

# Etc.

<details>
<summary>
<code>SHA</code> 확인하는 법
</summary>
<br />

```groovy Jenkinsfile
pipeline {
    agent any
    ...
    stages {
        stage() {
            steps {
                script {
                    def commitSha = sh(script: "git rev-parse HEAD", returnStdout: true).trim()
                    echo "Current commit SHA: ${commitSha}"
                }
            }
        }
    }
}
```

</details>
