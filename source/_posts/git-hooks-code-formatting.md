---
title: Git Hooks 기반 Python Code Formatting
date: 2023-07-07 16:13:43
categories:
- 3. DevOps
tags:
- CI/CD
- GitHub
- Python
---
# Introduction

[저번](https://zerohertz.github.io/automating-code-formatting-with-github-actions/#%EC%B0%A8%ED%9B%84-%EC%8B%9C%EB%8F%84)에 이어서 python code들의 formatting을 조금 더 간편하게 [Git Hooks](https://git-scm.com/book/ko/v2/Git%EB%A7%9E%EC%B6%A4-Git-Hooks)를 통해 도전한다.

## Git Hooks

Git Hooks는 Git 작업의 특정 지점에서 실행되는 스크립트다.
사용자 정의 작업을 수행하거나 작업의 유효성을 검사하기 위해 사용되며 git repository 내부에 설정되어 해당 이벤트가 발생할 때마다 실행된다.

## pre-commit

pre-commit은 Git Hooks을 활용하여 코드 commit 전에 자동으로 실행되는 도구다.
코드의 품질을 유지하고 일관성을 강제하기 위해 사용되며 일반적으로 코드 스타일 체크, 정적 분석, 테스트 실행 등의 작업을 수행한다.
또한 commit 하기 전에 코드에 대한 일련의 검사를 수행하여 품질을 향상시키고, 잠재적인 오류나 스타일 가이드 위반을 방지한다.

```shell
$ pip install pre-commit
```

pre-commit은 위와 같이 설치할 수 있으며, `.pre-commit-config.yaml` 파일을 사용하여 구성한다.
이 파일에는 사용할 Git Hooks 스크립트, 훅을 실행할 리포지토리 경로, 특정 파일에 대한 훅의 적용 여부 등의 설정이 포함된다.

pre-commit은 다양한 Git Hooks (코드 포맷팅, 정적 분석, 린팅, 테스트 실행 등)를 지원하며을 수행할 수 있다.
`.pre-commit-config.yaml` 파일에서 필요한 훅을 구성하고 해당 훅이 실행될 때 어떤 작업을 수행할지 결정할 수 있다.

<!-- More -->

---

# Python Code Formatting

```yaml .pre-commit-config.ymal
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/ambv/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/myint/autoflake
    rev: v2.2.0
    hooks:
      - id: autoflake
        args:
          - --in-place
          - --remove-unused-variables
          - --remove-all-unused-imports
          - --expand-star-imports
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

<details>
<summary>
<code>flake8</code> vs. <code>autoflake</code>
</summary>

<code>flake8</code>과 <code>autoflake</code>은 모두 Python 코드 정적 분석 도구입니다. 그러나 각각의 동작과 목적은 약간 다릅니다.

<code>flake8</code>:
<code>flake8</code>은 Python 코드의 문제를 식별하고 보고하는 데 사용되는 도구입니다. PEP 8 스타일 가이드 준수, 문법 오류, 네이밍 규칙 위반, 코드 복잡도 등과 같은 다양한 문제를 검사합니다. <code>flake8</code>은 Pycodestyle, PyFlakes 및 McCabe를 결합한 것으로, 코드 스타일, 잠재적인 오류 및 복잡도 문제를 포착할 수 있습니다. <code>flake8</code>은 코드의 가독성과 유지 관리 가능성을 향상시키는 데 도움이 됩니다.

<code>autoflake</code>:
<code>autoflake</code>은 <code>flake8</code>과 유사한 목표를 가지고 있지만, 추가적으로 사용되지 않는 변수와 임포트 문을 자동으로 제거하여 코드를 최적화합니다. <code>autoflake</code>은 사용되지 않는 코드 요소를 제거함으로써 코드 베이스를 정리하고, 불필요한 부분을 제거하여 코드 크기를 줄이는 데 도움이 됩니다. 이는 더 깔끔하고 효율적인 코드를 작성하는 데 도움이 될 수 있습니다.

따라서, <code>flake8</code>은 주로 코드 스타일과 잠재적인 오류를 검사하는 데 사용되며, <code>autoflake</code>은 사용되지 않는 코드 요소를 자동으로 제거하여 코드를 최적화하는 데 사용됩니다. 두 도구는 모두 코드 품질을 향상시키는 데 도움이 되는데, 각각의 목적과 사용 사례에 따라 적합한 도구를 선택할 수 있습니다.
</details>
<br />

```shell
$ pre-commit install
$ git add .pre-commit-config.yaml
$ git commit -m "Add: .pre-commit-config.yaml"
$ git push origin main
```

잘 적용되었는지 확인하기 위해 오류가 이미 발생한 `test.py`와 `test.yaml`를 commit 해보자.

```python test.py
for i in range(3):
     print(i)

for i in range(3):
 print(i)

for i in range(3):
	print(i)
        
```

![commit_test.py](/images/git-hooks-code-formatting/251974060-e7a1f54c-8ea1-4250-aefe-cfdf70b7f895.png)

수많은 오류들이 발생했음을 확인할 수 있다.
그리고 아래와 같이 명령어를 실행하면 올바르지 않던 포맷 혹은 문법이 잘 고쳐졌음을 확인할 수 있다.

![pre-commit](/images/git-hooks-code-formatting/251974091-0a92e35d-4cb5-401f-a7f1-ecd801e6a277.png)

아래와 같이 깔끔한 코드를 commit할 수 있다.

```python test.py
for i in range(3):
    print(i)

for i in range(3):
    print(i)

for i in range(3):
    print(i)
```

![push](/images/git-hooks-code-formatting/251974403-3ce45c5e-3dc3-4cfc-b6ad-b9fb5b4c437f.png)

최종적으로 위와 같이 Git Hooks에 의해 변경된 파일들을 다시 add하고 commit 후 push하면 된다.

```shell
$ pre-commit run --all-files
```

![run-all-files](/images/git-hooks-code-formatting/251974560-77c9e373-dfe4-4316-99ac-50145c9abb32.png)

---

# Etc.


## 근본

> ChatGPT 선생님의 고견에 따르면 `black`이 근본이라고 합니다.

![ChatGPT](/images/git-hooks-code-formatting/251985070-227a8faf-99f2-42f9-a2db-85c397061b8e.png)

그리고 여러 formatter를 사용하면 당연히 충돌이 발생할 수 있으니 잘 알아보고 기용하는 것이 바람직하다.

<details>
<summary>
충돌 주의 ~
</summary>

<code>autoflake</code>, <code>black</code>, <code>autopep8</code>은 모두 Python 코드를 자동으로 포맷팅하고 개선하는 도구입니다. 각각의 도구는 코드를 일관된 스타일로 변경하고 가독성을 향상시키는 목적을 가지고 있습니다. 그러나 이 도구들은 서로 다른 규칙과 알고리즘을 사용하므로 충돌이 발생할 수 있습니다.

충돌이 발생할 수 있는 상황은 다음과 같습니다:

1. 동일한 파일에서 중복 포맷팅:
   + <code>autoflake</code>, <code>black</code>, <code>autopep8</code>를 모두 동시에 적용하면 동일한 파일에서 중복된 코드 포맷팅이 발생할 수 있습니다.
   + 이는 코드에 예상치 못한 변경을 일으킬 수 있으며, 코드의 일관성을 해치고 예상치 못한 동작을 유발할 수 있습니다.
2. 서로 다른 스타일 규칙:
   + <code>autoflake</code>, <code>black</code>, <code>autopep8</code>은 각각 독자적인 스타일 규칙을 가지고 있습니다.
   + 따라서, 한 줄의 코드에 대해 각 도구가 다른 결과를 생성할 수 있습니다. 이는 코드 포맷팅 결과의 일관성을 해칠 수 있고, 코드 리뷰나 협업 과정에서 혼란을 야기할 수 있습니다.
3. 잠재적인 버그:
   + 동시에 여러 도구를 사용할 경우, 각 도구의 버그나 잠재적인 문제가 복합적으로 발생할 수 있습니다.
   + 이는 예기치 않은 결과를 초래할 수 있으며, 코드의 신뢰성에 영향을 줄 수 있습니다.
 
충돌을 최소화하고 일관된 코드 포맷팅을 유지하기 위해서는 동시에 여러 도구를 사용하는 대신에 하나의 코드 포맷터를 선택하여 일관성을 유지하는 것이 좋습니다. 각 도구는 각각의 장점과 특징을 가지고 있으므로, 개발자는 자신의 프로젝트와 팀의 요구에 맞게 최적의 도구를 선택하고 사용하는 것이 중요합니다.

</details>

## Pylint

저번에 언급한 [`Pylint`](https://github.com/pylint-dev/pylint)를 `pre-commit`으로 사용하려면 아래와 같이 적용하면 된다.

```yaml .pre-commit-config.yaml
...
  - repo: https://github.com/pylint-dev/pylint
    rev: v2.17.4
    hooks:
      - id: pylint
        name: pylint
        entry: pylint
        language: system
        types: [python]
        args:
          [
            "-rn",
          ]
...
```