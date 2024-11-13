---
title: 'Jupyter Notebook in Raspberry Pi'
date: 2020-09-19 12:38:28
categories:
- Etc.
tags:
- Raspberry Pi
- Python
---
# Raspberry Pi 초기 설정

~~~ps
pi@raspberrypi:~ $ sudo apt-get update
pi@raspberrypi:~ $ sudo apt-get install python3-pip -y
pi@raspberrypi:~ $ sudo apt-get update -y
pi@raspberrypi:~ $ sudo apt-get install python3-venv
pi@raspberrypi:~ $ python3 -m venv zerohertz
pi@raspberrypi:~ $ source ~/zerohertz/bin/activate
(zerohertz) pi@raspberrypi:~ $ pip install --upgrade pip
(zerohertz) pi@raspberrypi:~ $ pip install jupyter notebook
(zerohertz) pi@raspberrypi:~ $ pip install ipykernel
(zerohertz) pi@raspberrypi:~ $ python -m ipykernel install --user --name=zerohertz
(zerohertz) pi@raspberrypi:~ $ jupyter notebook --generate-config
~~~

<!-- More -->

***

# Jupyter notebook 초기 설정

~~~ps
(zerohertz) pi@raspberrypi:~ $ cd .jupyter/
(zerohertz) pi@raspberrypi:~/.jupyter $ ipython
Python 3.7.3 (default, Jul 25 2020, 13:03:44)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from notebook.auth import passwd

In [2]: passwd()
Enter password:
Verify password:
Out[2]: 'argon2:$argon2id$v=19$m=10240,t=10,p=8$m+aqEGBeyldGR0GXSWzrgA$WZwA4udVMOnjOX27aYJaBA'

In [3]: exit()

(zerohertz) pi@raspberrypi:~/.jupyter $ nano jupyter_notebook_config.py
~~~

> nano 에디터를 통해 아래와 같이 수정한다.

![Jupyter](https://user-images.githubusercontent.com/42334717/93658730-32e3d400-fa79-11ea-80dd-ed5086acd002.png)

> 가상환경에서 deactivate 명령어로 나올 수 있다.

~~~ps
(zerohertz) pi@raspberrypi:~ $ deactivate
pi@raspberrypi:~ $
~~~

***

# Jupyter notebook 실행

~~~ps
pi@raspberrypi:~ $ source ~/zerohertz/bin/activate
(zerohertz) pi@raspberrypi:~ $ jupyter notebook
~~~

> 만약, 백그라운드에서 Jupyter notebook을 실행하고 싶다면 아래와 같이 실행한다.

~~~ps
pi@raspberrypi:~ $ source ~/zerohertz/bin/activate
(zerohertz) pi@raspberrypi:~ $ nohup jupyter notebook &
~~~

> 종료 시, 아래와 같이 실행한다.

~~~ps
(zerohertz) pi@raspberrypi:~ $ lsof nohup.out
COMMAND    PID      USER   FD   TYPE DEVICE SIZE/OFF   NODE NAME
jupyter-n 3008 zerohertz    1w   REG  179,2     1310 401054 nohup.out
jupyter-n 3008 zerohertz    2w   REG  179,2     1310 401054 nohup.out
(zerohertz) pi@raspberrypi:~ $ kill -9 3008(PID)
~~~