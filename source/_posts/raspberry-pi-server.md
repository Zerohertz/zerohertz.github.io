---
title: 라즈베리파이로 간단한 서버 만들기
date: 2020-09-18 20:29:40
categories:
- Etc.
tags:
- Raspberry Pi
---
# RASPBERRY PI OS 설치

[RASPBERRY PI OS](https://www.raspberrypi.com/software/)

> 위 사이트에서 프로그램을 받은 뒤, SD카드 연결 후 아래와 같이 WRITE하면 설치가 된다.

![raspbian](/images/raspberry-pi-server/raspbian.png)

<!-- More -->

***

# ssh 연결 허용

> RASPBERRY PI OS는 기본적으로 ssh가 disable이므로 설정을 바꿔줘야 한다. SD카드의 최상위 디렉토리에 확장자가 없는 ssh라는 파일을 생성하면 ssh가 enable이 된다.

![ssh-1](/images/raspberry-pi-server/ssh-1.png)

> 만약 WiFi를 통하여 라즈베리 파이를 이용할 예정이라면, 아래의 이름과 소스를 작성하여 최상위 디렉토리에 저장해야한다.

~~~cpp wpa_supplicant.conf
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
    ssid="WiFi 이름"
    psk="WiFi 비밀번호"
}
~~~

***

# 라즈베리 파이의 IP

> 라즈베리 파이에 연결된 공유기의 Gateway 주소로 접속 후, 라즈베리 파이의 IP를 알아낸다. Gateway는 아래의 코드를 입력하여 알 수 있다.

```shell
$ route get default
```

![ip](/images/raspberry-pi-server/ip.png)

***

# ssh 접속

> 위의 IP를 이용해 ssh로 접속할 수 있다.

![ssh-2](/images/raspberry-pi-server/ssh-2.png)

![ssh-3](/images/raspberry-pi-server/ssh-3.png)

***

# 고정 IP 할당

> WiFi 내에서 라즈베리 파이가 항상 같은 IP(192.168.123.123)를 갖도록 설정하였다.

![ip-config-1](/images/raspberry-pi-server/ip-config-1.png)

![ip-config-2](/images/raspberry-pi-server/ip-config-2.png)

> 이후 공유기 자체의 IP를 고정 IP로 설정을 바꿔준다. 또한 포트포워딩을 통해 내부 포트는 22로, 외부 포트는 자유로 설정하고 IP는 위에서 라즈베리 파이에 할당한 IP를 기입한다.

***

# 외부에서 ssh 접속

```shell
$ ssh pi@xxx.xxx.xxx.xxx(고정IP) -p(외부 포트)
```

> 만약 외부 포트를 22로 할당해주었다면, -p를 제외하고 실행해도 무방하다. 만약 에러가 날 경우 아래의 소스를 사용하는 것을 권한다.

```shell
ssh-keygen -R xxx.xxx.xxx.xxx(IP)
```

***

# 사용자 생성 및 삭제

> 사용자 생성은 아래와 같이 진행하면 된다.

~~~ps
pi@zerohertz-pi:~ $ sudo passwd root
New password:
Retype new password:
passwd: password updated successfully
pi@zerohertz-pi:~ $ sudo su
root@zerohertz-pi:/home/pi# adduser zerohertz
Adding user `zerohertz' ...
Adding new group `zerohertz' (1001) ...
Adding new user `zerohertz' (1001) with group `zerohertz' ...
Creating home directory `/home/zerohertz' ...
Copying files from `/etc/skel' ...
New password:
Retype new password:
passwd: password updated successfully
Changing the user information for zerohertz
Enter the new value, or press ENTER for the default
	Full Name []:
	Room Number []:
	Work Phone []:
	Home Phone []:
	Other []:
Is the information correct? [Y/n] y
root@zerohertz-pi:/home/pi# sudo adduser zerohertz sudo
Adding user `zerohertz' to group `sudo' ...
Adding user zerohertz to group sudo
Done.
root@zerohertz-pi:/home/pi# cat /etc/group | grep pi
adm:x:4:pi
dialout:x:20:pi
cdrom:x:24:pi
sudo:x:27:pi,zerohertz
audio:x:29:pi
video:x:44:pi
plugdev:x:46:pi
games:x:60:pi
users:x:100:pi
input:x:105:pi
netdev:x:109:pi
pi:x:1000:
spi:x:999:pi
i2c:x:998:pi
gpio:x:997:pi
root@zerohertz-pi:/home/pi# usermod -G adm,dialout,cdrom,sudo,audio,video,plugdev,games,users,input,netdev,spi,i2c,gpio zerohertz
root@zerohertz-pi:/home/pi# cat /etc/group | grep pi
adm:x:4:pi,zerohertz
dialout:x:20:pi,zerohertz
cdrom:x:24:pi,zerohertz
sudo:x:27:pi,zerohertz
audio:x:29:pi,zerohertz
video:x:44:pi,zerohertz
plugdev:x:46:pi,zerohertz
games:x:60:pi,zerohertz
users:x:100:pi,zerohertz
input:x:105:pi,zerohertz
netdev:x:109:pi,zerohertz
pi:x:1000:
spi:x:999:pi,zerohertz
i2c:x:998:pi,zerohertz
gpio:x:997:pi,zerohertz
root@zerohertz-pi:/home/pi# userdel -r -f pi
userdel: user pi is currently used by process 539
userdel: pi mail spool (/var/mail/pi) not found
root@zerohertz-pi:/home/zerohertz# su zerohertz
zerohertz@zerohertz-pi:~ $
~~~

> 위와 같이 새 사용자인 zerohertz로 로그인에 성공할 수 있음을 볼 수 있다. 또한, 새 사용자에게 모든 파일을 전송하고 싶다면 아래와 같이 실행하면 된다.

~~~ps
zerohertz@zerohertz-pi:/home $ sudo cp -a pi/. zerohertz/
zerohertz@zerohertz-pi:/home $ ls
pi  zerohertz
zerohertz@zerohertz-pi:/home $ cd zerohertz/
zerohertz@zerohertz-pi:~ $ ls -al
total 68
drwxr-xr-x 12 pi   pi   4096 Sep 19 09:36 .
drwxr-xr-x  4 root root 4096 Sep 19 17:32 ..
-rw-------  1 pi   pi   1484 Sep 19 17:37 .bash_history
-rw-r--r--  1 pi   pi    220 Aug 20 11:31 .bash_logout
-rw-r--r--  1 pi   pi   3523 Aug 20 11:31 .bashrc
drwxr-xr-x  2 pi   pi   4096 Aug 20 11:40 Bookshelf
drwx------  3 pi   pi   4096 Sep 19 04:56 .cache
drwx------  4 pi   pi   4096 Sep 19 07:56 .config
drwx------  3 pi   pi   4096 Aug 20 12:09 .gnupg
drwxr-xr-x  2 pi   pi   4096 Sep 19 09:36 .ipynb_checkpoints
drwxr-xr-x  5 pi   pi   4096 Sep 19 05:01 .ipython
drwx------  2 pi   pi   4096 Sep 19 05:09 .jupyter
drwxr-xr-x  5 pi   pi   4096 Sep 19 04:56 .local
-rw-r--r--  1 pi   pi    807 Aug 20 11:31 .profile
drwxr-xr-x  4 pi   pi   4096 Sep 19 09:35 python_tutorial
-rw-r--r--  1 pi   pi    180 Sep 19 09:32 .wget-hsts
drwxr-xr-x  7 pi   pi   4096 Sep 19 04:58 zerohertz
zerohertz@zerohertz-pi:~ $ cd ..
zerohertz@zerohertz-pi:/home $ cd pi
zerohertz@zerohertz-pi:/home/pi $ ls -al
total 68
drwxr-xr-x 12 pi   pi   4096 Sep 19 09:36 .
drwxr-xr-x  4 root root 4096 Sep 19 17:32 ..
-rw-------  1 pi   pi   1484 Sep 19 17:37 .bash_history
-rw-r--r--  1 pi   pi    220 Aug 20 11:31 .bash_logout
-rw-r--r--  1 pi   pi   3523 Aug 20 11:31 .bashrc
drwxr-xr-x  2 pi   pi   4096 Aug 20 11:40 Bookshelf
drwx------  3 pi   pi   4096 Sep 19 04:56 .cache
drwx------  4 pi   pi   4096 Sep 19 07:56 .config
drwx------  3 pi   pi   4096 Aug 20 12:09 .gnupg
drwxr-xr-x  2 pi   pi   4096 Sep 19 09:36 .ipynb_checkpoints
drwxr-xr-x  5 pi   pi   4096 Sep 19 05:01 .ipython
drwx------  2 pi   pi   4096 Sep 19 05:09 .jupyter
drwxr-xr-x  5 pi   pi   4096 Sep 19 04:56 .local
-rw-r--r--  1 pi   pi    807 Aug 20 11:31 .profile
drwxr-xr-x  4 pi   pi   4096 Sep 19 09:35 python_tutorial
-rw-r--r--  1 pi   pi    180 Sep 19 09:32 .wget-hsts
drwxr-xr-x  7 pi   pi   4096 Sep 19 04:58 zerohertz
~~~

> 아래의 과정을 통해 sudo 명령어 사용 시, 비밀번호를 사용하지 않을 수 있다.

~~~ps
zerohertz@zerohertz-pi:~ $ sudo cp /etc/sudoers.d/010_pi-nopasswd /etc/sudoers.d/010_zerohertz-nopasswd
zerohertz@zerohertz-pi:~ $ sudo chmod u+w /etc/sudoers.d/010_zerohertz-nopasswd
zerohertz@zerohertz-pi:~ $ sudo nano /etc/sudoers.d/010_zerohertz-nopasswd
zerohertz@zerohertz-pi:~ $ sudo chmod u-w /etc/sudoers.d/010_zerohertz-nopasswd
zerohertz@zerohertz-pi:~ $ sudo rm -vf /etc/sudoers.d/010_pi-nopasswd
removed '/etc/sudoers.d/010_pi-nopasswd'
~~~

[Reference1](https://blog.dalso.org/raspberry-pi/raspberry-pi-4/7891)
[Reference2](https://gist.github.com/JeremyIglehart/84251d8b6405eaa640d6546b2a1ae8bc)