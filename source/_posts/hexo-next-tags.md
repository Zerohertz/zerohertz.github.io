---
title: Tags in Hexo NexT Theme
date: 2024-11-16 10:01:27
categories:
  - Etc.
tags:
  - Hexo
---

# Quote

```jinja Quote https://hexo.io/docs/tag-plugins#Block-Quote
{% quote [author[, source]] [link] [source_link_title] %}
content
{% endquote %}
```

```jinja
<!-- markdownlint-disable -->
{% quote Seth Godin http://sethgodin.typepad.com/seths_blog/2009/07/welcome-to-island-marketing.html Welcome to Island Marketing %}
Every interaction is both precious and an opportunity to delight.
{% endquote %}
<!-- markdownlint-enable -->
```

<!-- markdownlint-disable -->

{% quote Seth Godin http://sethgodin.typepad.com/seths_blog/2009/07/welcome-to-island-marketing.html Welcome to Island Marketing %}
Every interaction is both precious and an opportunity to delight.
{% endquote %}

<!-- markdownlint-enable -->
<br/>

```jinja Centered Quote https://theme-next.js.org/docs/tag-plugins/#Centered-Quote
{% centerquote %}Something{% endcenterquote %}

{% cq %}Something{% endcq %}
```

```jinja
{% cq %}Something{% endcq %}
```

{% cq %}Something{% endcq %}

<!-- more -->

````markdown Code Block https://hexo.io/docs/tag-plugins#Code-Block
```[language] [title] [url] [link text] [additional options]
code snippet
```
````

---

# Video

```jinja Video https://theme-next.js.org/docs/tag-plugins/#Video
{% video url %}
```

```jinja
<!-- markdownlint-disable -->
{% video /images/hexo-next-tags/boonmosa.mp4 %}
<!-- markdownlint-enable -->
```

<!-- markdownlint-disable -->

{% video /images/hexo-next-tags/boonmosa.mp4 %}

<!-- markdownlint-enable -->

---

# Group Pictures

```jinja Group Pictures https://theme-next.js.org/docs/tag-plugins/group-pictures
{% grouppicture [number]-[layout] %}
{% endgrouppicture %}

{% gp [number]-[layout] %}
{% endgp %}
```

```jinja
{% gp 3-3 %}
![](/props/zerohertz-black-red-og.png)
![](/props/zerohertz-black-red-og.png)
![](/props/zerohertz-black-red-og.png)
{% endgp %}
```

{% gp 3-3 %}
![](/props/zerohertz-black-red-og.png)
![](/props/zerohertz-black-red-og.png)
![](/props/zerohertz-black-red-og.png)
{% endgp %}

---

# Mermaid

```yaml _config.yml
highlight:
  exclude_languages:
    - mermaid
```

````jinja Mermaid https://theme-next.js.org/docs/tag-plugins/mermaid
{% mermaid type %}
{% endmermaid %}

```mermaid
type

```
````

## Graph

````markdown
```mermaid
graph TD
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```
````

```mermaid
graph TD
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```

## Sequence Diagram

````markdown
```mermaid
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
```
````

```mermaid
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
```

## Gantt

````markdown
```mermaid
gantt
dateFormat YYYY-MM-DD
section Section
Completed :done, des1, 2014-01-06,2014-01-08
Active :active, des2, 2014-01-07, 3d
Parallel 1 : des3, after des1, 1d
Parallel 2 : des4, after des1, 1d
Parallel 3 : des5, after des3, 1d
Parallel 4 : des6, after des4, 1d
```
````

```mermaid
gantt
dateFormat YYYY-MM-DD
section Section
Completed :done, des1, 2014-01-06,2014-01-08
Active :active, des2, 2014-01-07, 3d
Parallel 1 : des3, after des1, 1d
Parallel 2 : des4, after des1, 1d
Parallel 3 : des5, after des3, 1d
Parallel 4 : des6, after des4, 1d
```

## Class Diagram

````markdown
```mermaid
classDiagram
Class01 <|-- AveryLongClass : Cool
<<interface>> Class01
Class09 --> C2 : Where am i?
Class09 --* C3
Class09 --|> Class07
Class07 : equals()
Class07 : Object[] elementData
Class01 : size()
Class01 : int chimp
Class01 : int gorilla
class Class10 {
  <<service>>
  int id
  size()
}
```
````

```mermaid
classDiagram
Class01 <|-- AveryLongClass : Cool
<<interface>> Class01
Class09 --> C2 : Where am i?
Class09 --* C3
Class09 --|> Class07
Class07 : equals()
Class07 : Object[] elementData
Class01 : size()
Class01 : int chimp
Class01 : int gorilla
class Class10 {
  <<service>>
  int id
  size()
}
```

## State Diagram

````markdown
```mermaid
stateDiagram
[*] --> Still
Still --> [*]
Still --> Moving
Moving --> Still
Moving --> Crash
Crash --> [*]
```
````

```mermaid
stateDiagram
[*] --> Still
Still --> [*]
Still --> Moving
Moving --> Still
Moving --> Crash
Crash --> [*]
```

## Pie

````markdown
```mermaid
pie
"Dogs" : 386
"Cats" : 85
"Rats" : 15
```
````

```mermaid
pie
"Dogs" : 386
"Cats" : 85
"Rats" : 15
```

---

# Note

```jinja Note https://theme-next.js.org/docs/tag-plugins/note
{% note [class] [no-icon] [summary] %}
Any content (support inline tags too).
{% endnote %}
```

```jinja
{% note %}
No Parameters
{% endnote %}
{% note default %}
Default
{% endnote %}
{% note primary %}
Primary
{% endnote %}
{% note info %}
Info
{% endnote %}
{% note success %}
Success
{% endnote %}
{% note warning %}
Warning
{% endnote %}
{% note danger %}
Danger
{% endnote %}
{% note summary %}
Summary
{% endnote %}
```

{% note %}
No Parameters
{% endnote %}
{% note default %}
Default
{% endnote %}
{% note primary %}
Primary
{% endnote %}
{% note info %}
Info
{% endnote %}
{% note success %}
Success
{% endnote %}
{% note warning %}
Warning
{% endnote %}
{% note danger %}
Danger
{% endnote %}
{% note summary %}
Summary
{% endnote %}

---

# Tabs

```jinja Tabs https://theme-next.js.org/docs/tag-plugins/tabs
{% tabs Unique name, [index] %}
<!-- tab [Tab caption] [@icon] -->
Any content (support inline tags too).
<!-- endtab -->
{% endtabs %}
```

````jinja
{% tabs unique %}
<!-- tab Python -->
```python main.py
print("Hello, World!")
```
<!-- endtab -->
<!-- tab Java -->
```java Main.java
System.out.print("Hello, World!")
```
<!-- endtab -->
<!-- tab Go -->
```go main.go
fmt.Print("Hello, World!")
```
<!-- endtab -->
{% endtabs %}
````

{% tabs unique %}

<!-- tab Python -->

```python main.py
print("Hello, World!")
```

<!-- endtab -->
<!-- tab Java -->

```java Main.java
System.out.print("Hello, World!")
```

<!-- endtab -->
<!-- tab Go -->

```go main.go
fmt.Print("Hello, World!")
```

<!-- endtab -->

{% endtabs %}

---

{% note References %}

- [Hexo: Tag Plugins](https://hexo.io/docs/tag-plugins)
- [NexT: Tag Plugins](https://theme-next.js.org/docs/tag-plugins/)

{% endnote %}
