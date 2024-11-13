---
title: GitHub Actions 기반 Formatting 자동화
date: 2023-07-05 08:58:00
categories:
- 3. DevOps
tags:
- CI/CD
- GitHub
- Python
---
# Introduction

Algorithm 문제를 풀고 [GitHub](https://github.com/Zerohertz/Algorithm)에서 관리하고 있었는데, 문제를 해결할 때 [replit](https://replit.com/languages/python3)을 초기에 많이 사용했었다.
하지만 현재 사용하고 있는 Visual Studio Code와 indentation의 규칙이 달라서 GitHub Actions로 모두 동일한 format을 가지도록 자동화하겠다.

[GitHub Actions](https://docs.github.com/ko/actions)는 구동 조건 ([CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/), 특정 branch에 push, ...)을 만족하면 준비된 스크립트를 자동적으로 실행해준다.
따라서 CI/CD를 포함하여 다양한 작업에 사용되고 있다. ([로또 자동화](https://github.com/Zerohertz/lottery-bot/actions)도 가능,,,)

<img width="1322" alt="I_WANNA_BE_RICH" src="https://github.com/Zerohertz/lottery-bot/assets/42334717/85fd91e0-4bbe-4882-983c-b9f2dbd8b7e0">

<!-- More -->

---

# Automating Code Formatting

시작하기 전에 구동하려는 repository 내에서 "Setting - Secrets and variables - Actions"로 들어가 아래와 같이 GitHub token 값을 공개되지 않도록 환경 변수로 지정한다.

<img width="1143" alt="Secrets" src="https://github.com/Zerohertz/Algorithm/assets/42334717/8a69aa93-98d4-4078-bbf2-5bbd82ad04d3">

`${YOUR_REPOSITORY_PATH}/.github/workflows/${YOUR_ACTIONS}.yml` 경로에 원하는 행동을 작성하고 repository에 push하면 조건에 맞춰 실행된다.

`actions/checkout@v3`로 현재 구동하려는 repository를 지정해주고 필요한 환경을 설정해준다. (Python, Formatting tools)
shell script를 개발하는 것과 비슷하게 조건문으로 `isort`와 `autopep8`를 구동하고 변동사항이 있다면 push 할 수 있도록 개발하였다.

```yaml Formatter.yml
name: Formatter

on:
  push:
    branches:
      - main

jobs:
  Formatter:
    name: Formatter
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      with:
        repository: Zerohertz/Algorithm
        token: ${{ secrets.GH_TOKEN }}
        path: ./

    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'

    - name: Install isort & autopep8
      run: |
        pip install isort autopep8

    - name: Check & Push
      run: |
        isort .
        autopep8 --recursive --in-place --aggressive .
        if [[ -n $(git diff) ]]; then
          echo "Code changes detected."
          git config --global user.name "Zerohertz"
          git config --global user.email "ohg3417@gmail.com"
          git config --global credential.helper store
          git add .
          git commit -m ":art: Style: Format"
          git push
        else
          echo "No code changes."
        fi
```

준비된 `Formatter.yml` 파일을 push하면 아래와 같이 잘 실행된다.

<img width="1437" alt=":tada:" src="https://github.com/Zerohertz/lottery-bot/assets/42334717/8003276a-d94a-49dc-9ce3-f825b7634f5c">

[변경 내역](https://github.com/Zerohertz/Algorithm/commit/e3e238aebfdece8e851040819773f9f04e8ebda7)을 확인해보면 모두 정갈하게 바뀐 것을 알 수 있다.

---

# 차후 시도

[Pylint](https://github.com/pylint-dev/pylint)를 사용하면 python 코드를 평가해준다.
해당 기능을 GitHub Actions에 추가하여 사용하면 귀여울 것 같다.

```shell
$ pylint main.py
main.py:1:0: C0114: Missing module docstring (missing-module-docstring)
main.py:11:0: C0103: Constant name "cost" doesn't conform to UPPER_CASE naming style (invalid-name)

-----------------------------------
Your code has been rated at 9.43/10
```

마지막으로, 현재 문제점은 GitHub Actions가 main branch에 push 될 때 구동되는데 여기서 formatting을 진행하고 변경사항이 있으면 다시 push 하고, `git add .`를 사용하는 것이다.
이 말은 재귀적으로 GitHub Actions를 호출하는 것이기 때문에 이런 것을 어떻게 바꿀 수 있을지 고민해야한다.
아마 [Git Hooks & Husky](https://blog.pumpkin-raccoon.com/85)를 사용하면 이런 문제를 해결할 수 있을 것 같다.