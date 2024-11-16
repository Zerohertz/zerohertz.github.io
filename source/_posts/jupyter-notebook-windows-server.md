---
title: Building a Windows Server Infrastructure for Data Analysis
date: 2023-01-11 02:38:19
categories:
- 5. Machine Learning
tags:
- Python
- PyTorch
- TensorFlow
---
# SSH 설정 및 포트포워딩

[SSH Setup](https://www.lainyzine.com/ko/article/how-to-run-openssh-server-and-connect-with-ssh-on-windows-10/)

윈도우 컴퓨터에서 위 설정을 모두 마쳤다면 컴퓨터와 연결된 공유기의 gateway에 접속하여 포트포워딩 진행

![port-forwarding](/images/jupyter-notebook-windows-server/port-forwarding.png)

+ ssh 사용을 위한 `22` 포트와 jupyter notebook 사용을 위한 `8888` 포트를 포워딩

~~~sh
ssh username@XXX.XXX.XXX.XXX -p {port}
~~~

<!-- More -->

***

# Anaconda 설치

[Anaconda](https://www.anaconda.com/)

위의 홈페이지에서 anaconda 설치 후 아래 환경 변수 추가 (ssh에서 `conda` 명령어를 쓰기 위함)

> 환경 변수 설정 (Path)

~~~
C:\Users\username\anaconda3
C:\Users\username\anaconda3\Library
C:\Users\username\anaconda3\Scripts
C:\Users\username\anaconda3\Library\bin
~~~

> 가상 환경 생성

~~~python
conda create -n pytorch_env python=3.7
conda activate pytorch_env

conda create -n tensorflow_env python=3.7
conda activate tensorflow_env
~~~

---

# Library 설치

~~~python pytorch_env
conda install -y jupyter
conda install -y scikit-learn
conda install -y pandas
conda install -y numpy
conda install -y matplotlib
conda install -y seaborn
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
~~~

~~~python tensorflow_env
conda install -y jupyter
conda install -y scikit-learn
conda install -y pandas
conda install -y numpy
conda install -y matplotlib
conda install -y seaborn
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow==2.10"
~~~

+ [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
+ [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)

---

# Jupyter Notebook 설정

~~~python
from notebook.auth import passwd
passwd()
exit()
~~~

~~~sh
jupyter notebook --generate-config
~~~

~~~python jupyter_notebook_config.py
c = get_config()
c.NotebookApp.allow_origin = '*'
c.NotebookApp.notebook_dir = '절대 위치'
c.NotebookApp.ip = '*'
c.NotebookApp.port = 8888
c.NotebookApp.password = '토큰'
c.NotebookApp.password_required = True
c.NotebookApp.open_browser = False
~~~

---

# 최종 확인

![results](/images/jupyter-notebook-windows-server/results.png)