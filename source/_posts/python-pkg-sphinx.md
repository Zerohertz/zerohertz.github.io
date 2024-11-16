---
title: Sphinx 기반 Python Package 문서화
date: 2023-10-12 18:57:39
categories:
- Etc.
tags:
- Python
---
# Introduction

개발을 하다보면 [이런](https://opencv-python.readthedocs.io/en/latest/#) 페이지를 한번쯤 들어가본적 있을 것이다.
위 페이지와 같이 개발자로서 코드를 작성하는 것만큼 중요한 것은 해당 코드를 잘 문서화하는 것이다.
특히, 오픈 소스 프로젝트나 여러 사람들과 협업을 진행하는 큰 프로젝트에서는 코드의 문서화가 더욱 중요해진다.
문서화는 다른 개발자들이 코드를 이해하고 사용하는 데 큰 도움을 줄 뿐만 아니라, 코드의 유지 및 관리도 훨씬 쉬워진다.

Python은 그 자체로 간결하고 읽기 쉬운 언어이지만, 복잡한 패키지나 프로젝트를 다루게 되면 적절한 문서화 없이는 그 구조와 기능을 파악하기가 어려울 수 있다.
이때 필요한 것이 바로 문서화 도구다.
그 중에서도 Python 커뮤니티에서 널리 사용되는 도구가 바로 `Sphinx`다.

`Sphinx`는 Python 문서화를 위한 강력한 도구로, 간단한 마크다운 형식의 문서를 정교한 HTML, PDF, ePub 등의 형식으로 변환해준다.
또한, 다양한 플러그인과 확장 기능을 지원하여 문서의 내용뿐만 아니라 디자인, 구조, 그리고 상호 작용까지도 사용자의 필요에 맞게 커스터마이즈할 수 있다.

따라서 이 글에서는 `Sphinx`를 사용하여 Python 패키지를 어떻게 문서화하는지 간략히 알아본다.

<!-- More -->

---

# Sphinx

```bash
~$ pip install sphinx sphinx-rtd-theme
~$ mkdir docs
~$ cd docs
~/docs$ sphinx-quickstart
```

위와 같이 필요한 라이브러리를 설치 하고 설정을 진행할 수 있다.
`sphinx-rtd-theme`는 위에서 언급한 페이지의 테마로 널리 쓰이고 있다.

![sphinx-quickstart](/images/python-pkg-sphinx/274485563-880d97fb-c5b6-4178-898f-28c3c94dcf69.png)

설정 시 프로젝트의 이름, 저자의 이름 등 몇가지 정보를 입력하면 아래와 같이 초기 디렉토리가 구성된다.

```bash
~/docs$ tree
.
├── build
├── make.bat
├── Makefile
└── source
    ├── conf.py
    ├── index.rst
    ├── _static
    └── _templates
```

그 중 `source/conf.py`의 `extensions`에 아래와 같은 변수들을 추가하고 앞서 설치한 `sphinx_rtd_theme` 테마도 설정해준다.

```python source/conf.py
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PROJECT_NAME"
copyright = "2023, Thomas"
author = "Thomas"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google Style Python Docstrings 사용 시 추가
]
add_module_names = False  # 문서에 클래스 및 함수를 표시할 때 경로 생략 시 추가

templates_path = ["_templates"]
exclude_patterns = []

language = "ko"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # 다양한 문서에서 사용하는 테마
html_static_path = ["_static"]
```

아래와 같은 명령어로 빌드 및 테스트를 진행할 수 있다.

```bash
~$ sphinx-apidoc -f -o docs/source/ ${PACKAGE_NAME}
~$ sed -i '/.. automodule::/a\   :private-members:' docs/source/*.rst  # _으로 시작하는 private members를 문서화 하려면 해당 라인 실행
~$ cd docs
~/docs$ make html
~/docs$ cd build/html
~/docs/build/html$ python -m http.server
```

여기서 문서화할 패키지 내 클래스 및 함수의 정의 부분 하단에 아래의 양식에 맞춰 주석을 작성해야 올바른 문서가 작성된다.
아래의 양식은 [Google Style Python Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html)를 사용하는 것이고, 이 외에도 여러 양식이 존재한다.

<details>
<summary>
여러 양식들
</summary>

1. **reStructuredText (reST)**: Sphinx의 기본 문서 포맷입니다. reST는 문서와 주석에서 사용할 수 있는 가볍고 읽기 쉬운 마크업 언어입니다.

2. **Numpydoc/NumPy Style**: Numpy 프로젝트에서 시작된 스타일로, 데이터 사이언스와 관련된 프로젝트에서 널리 사용됩니다. NumPy, SciPy, Matplotlib 등의 프로젝트에서 이 스타일을 볼 수 있습니다.

   예:
   ```python
   def function_with_types_in_docstring(param1, param2):
       """
       Example function with types documented in the docstring.

       `PEP 484`_ type annotations are supported. If attribute, parameter, and
       return types are annotated according to `PEP 484`_, they do not need to be
       included in the docstring:

       Parameters
       ----------
       param1 : int
           The first parameter.
       param2 : str
           The second parameter.

       Returns
       -------
       bool
           True if successful, False otherwise.
       """
   ```

3. **Google Style Python Docstrings**: Google에서 사용하는 스타일로, 코드의 가독성을 높이기 위해 설계되었습니다. Google의 Python 스타일 가이드에 따르면 이 스타일로 주석을 작성하게 됩니다.

   예:
   ```python
   def sample_function(param1, param2):
       """
       Summary line.

       Extended description of function.

       Args:
       param1 (int): Description of param1.
       param2 (str): Description of param2.

       Returns:
       bool: Description of return value.
       """
   ```

4. **Epytext Style**: Epydoc에서 사용하는 스타일로, `@` 기호를 사용하여 인수, 반환 타입, 예외 등을 주석에 기술합니다.

</details>

```python
"""This is a function example with Google style docstrings.

The main description of the function is written here and provides
a brief overview of what the function does.

Args:
    param1 (int): The first parameter used for ...
    param2 (str): The second parameter used for ...

Returns:
    bool: True if successful, False otherwise.

Raises:
    TypeError: If `param1` is not an integer.
    ValueError: If `param2` is an empty string.

Notes:
    This is an additional note or a set of notes that provide extra
    information about the function, but are not part of the main description.

Examples:
    >>> example_function(5, "hello")
    True
    >>> example_function(10, "")
    ValueError: param2 should not be an empty string.
"""
```

+ 첫 번째 줄
  + 클래스 또는 메서드의 짧고 간결한 요약 제공 (한 줄 요약)
  + 함수 또는 메서드의 리스트를 보여줄 때 사용
  + 짧고 간결해야 하며, 종종 도트(.)로 끝나기도 함
+ 두 번째 줄 이후
  + 클래스 또는 메서드에 대한 보다 상세한 설명 제공
  + 함수의 작동 방식, 내부 로직, 사용 사례 등에 대한 추가적인 상세 정보 포함
  + 여러 문장이나 단락으로 구성
+ `Args`
  + 함수의 인수를 설명
  + 각 인수는 이름, 유형 및 설명으로 구성
+ `Returns`
  + 함수의 반환 값 설명
+ `Raises`
  + 함수가 발생시킬 수 있는 예외 나열
+ `Notes`
  + 추가 정보나 설명 제공
+ `Examples`
  + 함수의 사용 예

---

# Examples

```python sum.py
def sum(arg1, arg2):
    """
    This is sum function.

    Args:
        arg1 (int): Description of arg1.
        arg2 (int): Description of arg2.

    Returns:
        int: Description of return value

    Examples:
        >>> a=1
        >>> b=2
        >>> sum(a, b)
        3
    """
    return arg1 + arg2
```

![sum](/images/python-pkg-sphinx/274756106-c1553d5b-041f-4739-a4b9-670a67db45ea.png)

```python example_function.py
def example_function(param1, param2):
    """This is a function example with Google style docstrings.

    The main description of the function is written here and provides
    a brief overview of what the function does.

    Args:
        param1 (int): The first parameter used for ...
        param2 (str): The second parameter used for ...

    Returns:
        bool: True if successful, False otherwise.

    Raises:
        TypeError: If `param1` is not an integer.
        ValueError: If `param2` is an empty string.

    Notes:
        This is an additional note or a set of notes that provide extra
        information about the function, but are not part of the main description.

    Examples:
        >>> example_function(5, "hello")
        True
        >>> example_function(10, "")
        ValueError: param2 should not be an empty string.
    """
    if not isinstance(param1, int):
        raise TypeError("param1 should be of type int")
    if not param2:
        raise ValueError("param2 should not be an empty string")

    # Function's logic here
    return True
```

![example_function](/images/python-pkg-sphinx/274760103-e7b50f3f-1c44-44e3-adc0-4933fbf96fdc.png)

---

# Plugins

|Name|Features|
|:-:|:-:|
|`sphinx.ext.autodoc`|자동으로 Python 모듈로부터 문서를 생성합니다. 클래스, 함수, 모듈 등의 주석을 사용하여 문서화합니다.|
|`sphinx.ext.coverage`|문서 커버리지를 체크하는 도구입니다. 문서화되지 않은 코드 부분을 찾아내는 데 사용됩니다.|
|`sphinx.ext.doctest`|문서 내에 포함된 doctest를 테스트하고 실행합니다. 이를 통해 문서의 예제 코드가 정확한지 검증할 수 있습니다.|
|`sphinx.ext.duration`|문서화 프로세스의 각 단계가 얼마나 걸리는지 기록합니다. 성능 분석과 최적화에 유용합니다.|
|`sphinx.ext.githubpages`|생성된 문서를 GitHub Pages에 쉽게 배포할 수 있도록 도와줍니다.|
|`sphinx.ext.intersphinx`|다른 Sphinx 문서와 링크를 연결할 수 있게 해주는 확장 기능입니다. 서로 다른 문서 간의 참조를 가능하게 합니다.|
|`sphinx.ext.mathjax`|문서 내의 수학적 표현을 렌더링하기 위해 MathJax를 사용합니다. 복잡한 수학 공식의 표현을 지원합니다.|
|`sphinx.ext.napoleon`|Google 스타일과 NumPy 스타일의 docstring을 지원합니다. 코드 문서화에 일관성을 제공합니다.|
|`sphinx.ext.todo`|문서 내의 TODO 항목들을 관리하고 표시합니다. 개발 중인 부분이나 미완성 부분을 표시하는 데 유용합니다.|
|`sphinxcontrib.gtagjs`|Google 태그 관리자를 사용하여 Sphinx 문서에 분석 툴을 추가합니다. 웹사이트 트래픽과 상호작용을 추적할 수 있습니다.|
|`sphinxcontrib.jquery`|jQuery 라이브러리를 Sphinx 문서에 통합합니다. 동적인 웹 컨텐츠와 인터랙션을 구현할 수 있습니다.|
|`sphinxext.opengraph`|Open Graph 메타 데이터를 Sphinx 문서에 추가하여, 소셜 미디어에서 더 나은 공유를 가능하게 합니다.|
|`sphinx_copybutton`|코드 블록에 복사 버튼을 추가하여 사용자가 코드를 쉽게 복사할 수 있게 합니다.|
|`sphinx_favicon`|사용자 정의 파비콘을 Sphinx 문서에 추가할 수 있습니다. 웹사이트의 브랜딩을 강화하는 데 도움이 됩니다.|
|`sphinx_paramlinks`|문서 내의 파라미터 참조를 링크로 변환하여, 파라미터 간의 쉬운 탐색을 지원합니다.|
|`myst_parser`|Markdown 문법을 사용하여 Sphinx 문서를 작성할 수 있게 해주는 파서입니다. Markdown과 reStructuredText를 혼용할 수 있습니다.|