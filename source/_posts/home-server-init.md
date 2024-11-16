---
title: Home Server 구축기
date: 2023-08-03 22:19:31
categories:
- 0. Daily
tags:
- Home Server
---
# Introduction

[이 영상](https://www.youtube.com/watch?v=JFTUzjW0fsI&ab_channel=ITSub%EC%9E%87%EC%84%AD)을 보며 뽐뿌가 온 나머지,,, 홈 서버로 사용하기 위해 아래와 같은 스펙으로 주문했다!!!

|Type|Link|Price|
|:-:|:-:|:-:|
|Case|[ASRock DeskMini X300 120W](https://link.coupang.com/a/5MYlc)|215,000원|
|CPU|[AMD Ryzen 5 5600G](https://smartstore.naver.com/sjsys/products/8848858024?NaPm=ct%3Dlkv6wk2g%7Cci%3Dcheckout%7Ctr%3Dppc%7Ctrx%3Dnull%7Chk%3D50844c3aabed4ccb245016a75376a405e62edc4c) + Thermal grease|166,500원|
|Fan|[Noctua NH-L9a-AM4](https://link.coupang.com/a/5MYDl)|75,990원|
|RAM|[SAMSUNG Notebook DDR4 16GB PC4-25600](https://shopping.interpark.com/product/productInfo.do?prdNo=11350045170) $\times$ 2|77,860원|
|SSD|[SK Hynix GOLD P31 NVMe SSD M.2 NVMe 1TB](https://smartstore.naver.com/modeit/products/5356661216?NaPm=ct%3Dlkv6y0j3%7Cci%3Dcheckout%7Ctr%3Dppc%7Ctrx%3Dnull%7Chk%3Da5fb5d797cc979a1fcebe48762ff6de9a3aa26f1)|87,500원|

총 622,850원으로 구성할 수 있었다.

<!-- More -->

---

# 조립기

![1](/images/home-server-init/258575015-05a7c640-a760-49f5-85b8-50f2d6426047.png)

먼저 본체 후면에 존재하는 4개의 나사를 풀어 위 사진과 같이 메인보드를 꺼낸다.

![2](/images/home-server-init/258575023-64a4ce2c-4ce7-48e5-b4e8-c4e6cde8fb65.png)

CPU를 방향에 맞게 조립한다.

![3](/images/home-server-init/258575027-443a8534-46c6-4d2c-ad09-5b9097e80f9e.png)

RAM, SSD를 조립하고 쿨러를 장착하기 전에 thermal grease를 도포한다.
마지막으로 팬을 장착하면!

![4](/images/home-server-init/258575028-329d203e-7712-441a-b419-90410c2fae68.png)

완성이다.

---

# 구축기

## Ubuntu Setup

[여기](https://ubuntu.com/server)에서 다운로드 받은 이미지를 [rufus](https://rufus.ie/ko/)로 아래와 같이 booting USB를 만들면 된다.

![Ubuntu](/images/home-server-init/258264720-7cc48a80-a040-47fd-ab47-a03e04fddc6f.png)

그리고 USB를 연결하고 부팅한 뒤 기본 설정을 진행하는데, 아래와 같이 OpenSSH 패키지를 함께 설치한다.

![SSH](/images/home-server-init/258575044-4d14daa6-e584-42b8-ac84-e886973b2e23.png)

이후 추가적인 설정을 마치고 재부팅하면 모든 설치가 완료된다.
혹시 조립이 잘못됐는지 확인하기 위해 `df -h`와 `htop`을 실행하여 확인해본다.

![df_htop](/images/home-server-init/258575033-809a0028-4d40-4afe-9500-be808ae8ba00.png)

~~SSD, CPU, RAM 모두 잘 장착됐다!~~는 SSD 파티션이 놀고있다.
아래와 같이 놀고있는 파티션을 확장해준다.

```shell
$ df -hT
Filesystem                        Type   Size  Used Avail Use% Mounted on
tmpfs                             tmpfs  2.8G  1.4M  2.8G   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv ext4    98G   12G   82G  13% /
tmpfs                             tmpfs   14G     0   14G   0% /dev/shm
tmpfs                             tmpfs  5.0M     0  5.0M   0% /run/lock
/dev/nvme0n1p2                    ext4   2.0G  131M  1.7G   8% /boot
/dev/nvme0n1p1                    vfat   1.1G  6.1M  1.1G   1% /boot/efi
tmpfs                             tmpfs  2.8G  4.0K  2.8G   1% /run/user/1000
$ lsblk
NAME                      MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0                       7:0    0  63.3M  1 loop /snap/core20/1822
loop1                       7:1    0  63.4M  1 loop /snap/core20/1974
loop2                       7:2    0 111.9M  1 loop /snap/lxd/24322
loop3                       7:3    0  49.8M  1 loop /snap/snapd/18357
loop4                       7:4    0  53.3M  1 loop /snap/snapd/19457
nvme0n1                   259:0    0 931.5G  0 disk 
├─nvme0n1p1               259:1    0     1G  0 part /boot/efi
├─nvme0n1p2               259:2    0     2G  0 part /boot
└─nvme0n1p3               259:3    0 928.5G  0 part 
  └─ubuntu--vg-ubuntu--lv 253:0    0   100G  0 lvm  /
$ sudo lvscan
  ACTIVE            '/dev/ubuntu-vg/ubuntu-lv' [100.00 GiB] inherit
$ sudo lvextend -l +100%FREE -n /dev/ubuntu-vg/ubuntu-lv
$ lsblk
NAME                      MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0                       7:0    0  63.3M  1 loop /snap/core20/1822
loop1                       7:1    0  63.4M  1 loop /snap/core20/1974
loop2                       7:2    0 111.9M  1 loop /snap/lxd/24322
loop3                       7:3    0  49.8M  1 loop /snap/snapd/18357
loop4                       7:4    0  53.3M  1 loop /snap/snapd/19457
nvme0n1                   259:0    0 931.5G  0 disk 
├─nvme0n1p1               259:1    0     1G  0 part /boot/efi
├─nvme0n1p2               259:2    0     2G  0 part /boot
└─nvme0n1p3               259:3    0 928.5G  0 part 
  └─ubuntu--vg-ubuntu--lv 253:0    0 928.5G  0 lvm  /
$ sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
$ df -hT
Filesystem                        Type   Size  Used Avail Use% Mounted on
tmpfs                             tmpfs  2.8G  1.4M  2.8G   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv ext4   914G   12G  864G   2% /
tmpfs                             tmpfs   14G     0   14G   0% /dev/shm
tmpfs                             tmpfs  5.0M     0  5.0M   0% /run/lock
/dev/nvme0n1p2                    ext4   2.0G  131M  1.7G   8% /boot
/dev/nvme0n1p1                    vfat   1.1G  6.1M  1.1G   1% /boot/efi
tmpfs                             tmpfs  2.8G  4.0K  2.8G   1% /run/user/1000
$ sh sh/storage.sh 
전체 용량:      935.367 GB
사용 용량:      11.3012 GB
남은 용량:      885.777 GB
1.20820589851244350758 %
```

마지막에 사용된 `storage.sh`는 아래와 같다.

```bash sh/storage.sh
df -P | grep -v ^Filesystem | awk '{sum += $2} END { print "전체 용량:\t" sum/1024/1024 " GB" }'
df -P | grep -v ^Filesystem | awk '{sum += $3} END { print "사용 용량:\t" sum/1024/1024 " GB" }'
df -P | grep -v ^Filesystem | awk '{sum += $4} END { print "남은 용량:\t" sum/1024/1024 " GB" }'

DISK_TOTAL=`df -P | grep -v ^Filesystem | awk '{sum += $2} END { print sum; }'`
DISK_USED=`df -P | grep -v ^Filesystem | awk '{sum += $3} END { print sum; }'`
DISK_PER=`echo "100*$DISK_USED/$DISK_TOTAL" | bc -l`
echo "$DISK_PER %"
```

로그를 확인할 때 헷갈릴 수 있으니 아래와 같이 서버의 시간을 변경한다.

```shell
$ sudo timedatectl set-timezone Asia/Seoul
$ timedatectl
```

업데이트를 하고 재부팅을 한번 해준다.

```shell
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo reboot
```

## SSH Setup

외부에서 사용할 수 있게 port forwarding을 할 예정인데, 혹시 모를 의문의 중국 해커를 위해 방화벽을 세워보자!
그 전에 SSH의 port를 변경한다.

```shell
$ sudo vi /etc/ssh/sshd_config
...
Include /etc/ssh/sshd_config.d/*.conf

Port ${SSH_Port}
...
```

이렇게 파일을 변경하고 `service ssh restart`를 실행하면 적용된다.

### UFW (Uncomplicated Firewall)

> UFW: 리눅스 시스템에서 간단하게 방화벽을 설정하고 관리할 수 있는 도구

```shell
$ sudo apt-get install ufw
$ sudo ufw status verbose            # UFW 상태 확인
Status: inactive                     # 아직 실행을 하지 않아 비활성화 상태
$ sudo ufw default deny incoming
Default incoming policy changed to 'deny'
(be sure to update your rules accordingly)
$ sudo ufw default allow outgoing
Default outgoing policy changed to 'allow'
(be sure to update your rules accordingly)
$ sudo ufw allow ${SSH_Port}         # SSH
Rules updated
Rules updated (v6)
$ sudo ufw allow from ${Client_IP}   # Client
Rules updated
$ sudo ufw enable
Command may disrupt existing ssh connections. Proceed with operation (y|n)? y
Firewall is active and enabled on system startup
$ sudo ufw status verbose
Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing), disabled (routed)
New profiles: skip

To                         Action      From
--                         ------      ----
${SSH_Port}                ALLOW IN    Anywhere                  
Anywhere                   ALLOW IN    ${Client_IP}              
${SSH_Port} (v6)           ALLOW IN    Anywhere (v6)       
```

|Command|Mean|
|:-:|:-:|
|`sudo ufw enable`|UFW 활성화 및 부팅 시 자동 시작 설정|
|`sudo ufw disable`|UFW 비활성화|
|`sudo ufw status`|현재 방화벽 상태 표시|
|`sudo ufw allow`|특정 포트나 서비스 허용|
|`sudo ufw deny`|특정 포트나 서비스 차단|
|`sudo ufw delete`|규칙 삭제|
|`sudo ufw reset`|방화벽 설정 초기화|
|`sudo ufw reload`|설정 재로드|
|`sudo ufw default`|기본 정책 설정|
|`sudo ufw logging`|로그 기록 설정|
|`sudo ufw app list`|사용 가능한 응용 프로그램 리스트 표시|
|`sudo ufw app allow`|특정 응용 프로그램 허용|
|`sudo ufw app deny`|특정 응용 프로그램 차단|
|`sudo ufw limit`|연결 제한 설정 (예: `sudo ufw limit ssh/tcp`)|

### Fail2Ban

> Fail2Ban: 불법적인 로그인 시도와 같은 악의적인 행위로부터 시스템을 보호하기 위해 사용되는 오픈 소스 소프트웨어로 주로 리눅스와 유닉스 시스템에서 사용되며, 불법적인 활동을 탐지하고 이를 차단하여 시스템의 보안 강화

+ 로그 모니터링
  + 로그 파일을 모니터링하여 시스템에 대한 악성 행위 감지
  + SSH 로그인 시도 실패, 웹 서버의 404 에러 등 모니터링
+ 악의적 행위 탐지
  + 설정된 규칙에 따라서 로그 파일을 분석하여 여러 악의적 행위 식별
  + 로그인 실패, 대량의 접속 시도, 금지된 페이지 액세스 등
+ 임시 차단
  + 악성 행위를 감지 시 해당 IP 주소 일시적 차단
+ 다양한 서비스 지원
  + 서비스별로 설정 파일을 통해 각각의 동작 조정
  + SSH, Apache, Nginx 등 여러 서비스 보호 가능
+ 유연한 설정
  + 설정 파일을 사용하여 규칙, 차단 시간, 차단 임계치 등 조정

```shell
$ sudo apt-get install fail2ban
$ sudo service fail2ban status
○ fail2ban.service - Fail2Ban Service
     Loaded: loaded (/lib/systemd/system/fail2ban.service; disabled; vendor preset: enabled)
     Active: inactive (dead)
       Docs: man:fail2ban(1)
$ sudo systemctl enable fail2ban
$ sudo systemctl start fail2ban
$ sudo service fail2ban status
● fail2ban.service - Fail2Ban Service
     Loaded: loaded (/lib/systemd/system/fail2ban.service; enabled; vendor preset: enabled)
     Active: active (running) since Sat 2023-08-05 19:26:30 KST; 19s ago
       Docs: man:fail2ban(1)
   Main PID: 1343 (fail2ban-server)
      Tasks: 5 (limit: 33386)
     Memory: 15.4M
        CPU: 96ms
     CGroup: /system.slice/fail2ban.service
             └─1343 /usr/bin/python3 /usr/bin/fail2ban-server -xf start

Aug 05 19:26:30 0hz systemd[1]: Started Fail2Ban Service.
Aug 05 19:26:30 0hz fail2ban-server[1343]: Server ready
```

설정을 변경하려면 아래와 같이 진행한다. ([여기](https://nitr0.tistory.com/328) 참고!)
`cat /etc/fail2ban/jail.conf`을 통해 설정에 대한 설명을 확인할 수 있다.

```shell
$ sudo vi /etc/fail2ban/jail.local
$ service fail2ban restart
```

+ `sudo fail2ban-client status sshd`: 감옥 간 IP 목록 확인
+ `sudo fail2ban-client set sshd unbanip ${IP}`: 특정 IP 석방
+ `sudo fail2ban-client set sshd banip ${IP}`: 특정 IP 다시 감옥 보내기

### Google OTP

이렇게까지 해야하나 싶지만 어떤 분은 Amazon EC2를 털려 220만원을 냈다고 한다... (무서워용 ~)

```shell
$ sudo apt-get install libpam-google-authenticator
$ google-authenticator
```

위 명령어들을 실행하면 QR 코드가 반겨준다.
이걸 [Google Autheticator](https://apps.apple.com/us/app/google-authenticator/id388497605)에 등록하면 된다.
또한 서버에 SSH에 Google OTP를 추가하기 위해 아래 추가 및 변경을 진행한다.

```bash /etc/pam.d/sshd
...
auth required pam_google_authenticator.so
```

```bash /etc/ssh/sshd_config
...
ChallengeResponseAuthentication yes     # Ubuntu 20.04 LTS
KbdInteractiveAuthentication yes        # Ubuntu 22.04 LTS
...
```

`sudo service sshd restart` 수행 후 아래와 같이 사용자의 비밀번호와 OTP의 인증번호를 요구하는 것을 확인할 수 있다.

```shell
$ ssh ${User_Name}@${Server_IP} -p ${Port}
(${User_Name}@${Server_IP}) Password: 
(${User_Name}@${Server_IP}) Verification code: 
```

~~보안은 이제 그만.~~

### Client

서버에 접속할 개인 환경에서 `cat ~/.ssh/id_rsa.pub`를 실행 후 출력된 공개키를 서버의 `~/.ssh/authorized_keys`에 등록하고, `~/.ssh/config`에 아래와 같이 파일을 변경하면 비밀번호와 OTP 없이 바로 접속할 수 있다.

```bash
Host ${Host_Name}
    HostName ${IP_or_DDNS}
    User ${User_Name}
    Port ${SSH_Port_Num}
    IdentityFile ~/.ssh/id_rsa
```

```shell
$ ssh ${Host_Name}
```

## ZSH

SSH로 접속하고 계속 코딩하고 싶게 만들어줄 테마를 설치하기 위해 아래 명령어들을 실행한다.

```shell
$ sudo apt install zsh
$ chsh -s $(which zsh)
# 재접속!
This is the Z Shell configuration function for new users,
zsh-newuser-install.
You are seeing this message because you have no zsh startup files
(the files .zshenv, .zprofile, .zshrc, .zlogin in the directory
~).  This function can help you with a few settings that should
make your use of the shell easier.

You can:

(q)  Quit and do nothing.  The function will be run again next time.

(0)  Exit, creating the file ~/.zshrc containing just a comment.
     That will prevent this function being run again.

(1)  Continue to the main menu.

(2)  Populate your ~/.zshrc with the configuration recommended
     by the system administrator and exit (you will need to edit
     the file by hand, if so desired).

--- Type one of the keys in parentheses --- 2
/home/zerohertz/.zshrc:15: scalar parameter HISTFILE created globally in function zsh-newuser-install
(eval):1: scalar parameter LS_COLORS created globally in function zsh-newuser-install
```

```shell
$ sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
$ git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
$ git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
$ git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

```zsh ~/.zshrc
...
ZSH_THEME="powerlevel10k/powerlevel10k"
...
plugins=(
    git
    docker
    zsh-syntax-highlighting
    zsh-autosuggestions
)
...
```

```shell
$ source ~/.zshrc
```

이후에 아래와 같이 설정 선택을 마치면 아래와 같이 바뀐 테마를 확인할 수 있다.

![10k](/images/home-server-init/258623263-a3f3aeb7-560d-4762-9d61-8f7492e96c7c.gif)

![zsh](/images/home-server-init/258618231-7256b609-298f-45bd-a21b-f3e0167549bf.png)

다시 설정을 원할 경우 `p10k configure`를 실행하면 된다.
또한 Visual Studio Code에서 사용할 때 색을 변경하려면 아래와 같이 `settings.json`을 변경하면 된다.

```json settings.json
{
    ...
    "workbench.colorCustomizations": {
        "terminal.background":"#000000",
        "terminal.foreground":"#FFFFFF",
        "terminalCursor.background":"#FFFFFF",
        "terminalCursor.foreground":"#FFFFFF",
        "terminal.ansiBlack":"#000000",
        "terminal.ansiBlue":"#800A0A",
        "terminal.ansiBrightBlue":"#00FFFF",
        "terminal.ansiBrightCyan":"#0000FF",
        "terminal.ansiBrightGreen":"#00FF00",
        "terminal.ansiBrightMagenta":"#FF00F0",
        "terminal.ansiBrightRed":"#FF0000",
        "terminal.ansiBrightWhite":"#F7F7F7",
        "terminal.ansiBrightYellow":"#FFFF00",
        "terminal.ansiCyan":"#00FFFF",
        "terminal.ansiGreen":"#00FF00",
        "terminal.ansiMagenta":"#FF00F0",
        "terminal.ansiRed":"#FF0000",
        "terminal.ansiWhite":"#FFFFFF",
        "terminal.ansiYellow":"#FFFF00",
    ...
    },
    ...
}
```

---

# Etc.

1. `journalctl -f`: 서버의 SSH 로그인 로그를 확인
2. `speedtest-cli`: 서버의 인터넷 속도 측정

## Hardware Info

```shell
$ sudo apt-get install lshw
$ sudo lshw -html > hardware.html
```

위 명령어들로 생성한 `hardware.html`을 web browser로 출력하면 서버의 hardware 정보들을 확인할 수 있다.
이 외에도 아래 명령어들로 개별적인 hardware의 정보를 파악할 수 있다.

+ `lscpu`: CPU 정보
+ `lsblk`: Hard drive 및 partition 정보
+ `lspci`: PCI bus에 연결된 장치 정보
+ `lsusb`: USB bus에 연결된 장치 정보
+ `free -h`: System의 memory 사용 및 여유 공간 정보

## Hardware Monitoring

```shell
$ sudo apt install lm-sensors
$ sudo sensors-detect
$ sensors
nct6793-isa-0290
Adapter: ISA adapter
in0:                   264.00 mV (min =  +0.00 V, max =  +1.74 V)
in1:                     1.89 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in2:                     3.42 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in3:                     3.42 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in4:                   248.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in5:                   128.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in6:                   824.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in7:                     3.41 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in8:                     3.30 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in9:                     1.84 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in10:                  176.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in11:                  128.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
in12:                    1.89 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in13:                    1.73 V  (min =  +0.00 V, max =  +0.00 V)  ALARM
in14:                  192.00 mV (min =  +0.00 V, max =  +0.00 V)  ALARM
fan1:                     0 RPM  (min =    0 RPM)
fan2:                  1363 RPM  (min =    0 RPM)
fan3:                     0 RPM  (min =    0 RPM)
fan4:                     0 RPM  (min =    0 RPM)
fan5:                     0 RPM  (min =    0 RPM)
SYSTIN:                +118.0°C  (high =  +0.0°C, hyst =  +0.0°C)  sensor = thermistor
CPUTIN:                 +46.0°C  (high = +80.0°C, hyst = +75.0°C)  sensor = thermistor
AUXTIN0:                +35.0°C  (high =  +0.0°C, hyst =  +0.0°C)  ALARM  sensor = thermistor
AUXTIN1:               +110.0°C    sensor = thermistor
AUXTIN2:               +109.0°C    sensor = thermistor
AUXTIN3:               +109.0°C    sensor = thermistor
SMBUSMASTER 0:          +35.5°C  
PCH_CHIP_CPU_MAX_TEMP:   +0.0°C  
PCH_CHIP_TEMP:           +0.0°C  
PCH_CPU_TEMP:            +0.0°C  
intrusion0:            OK
intrusion1:            ALARM
beep_enable:           disabled

nvme-pci-0100
Adapter: PCI adapter
Composite:    +41.9°C  (low  =  -0.1°C, high = +82.8°C)
                       (crit = +83.8°C)
Sensor 1:     +34.9°C  (low  = -273.1°C, high = +65261.8°C)
Sensor 2:     +39.9°C  (low  = -273.1°C, high = +65261.8°C)

k10temp-pci-00c3
Adapter: PCI adapter
Tctl:         +35.9°C  

amdgpu-pci-0300
Adapter: PCI adapter
vddgfx:      731.00 mV 
vddnb:       937.00 mV 
edge:         +34.0°C  
slowPPT:       2.00 mW
```

+ `k10temp-pci-00c3`: CPU 온도
+ `nvme-pci-0100`: SSD 온도
+ `amdgpu-pci-0300`: GPU 온도

## `cat`

```shell
$ sudo apt install bat 
$ ln -s /usr/bin/batcat ~/.local/bin/bat
$ cat ContinuousTraining.py
import airflow
from airflow.decorators import dag
from airflow.operators.python_operator import PythonOperator
...
$ bat ContinuousTraining.py
───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       │ File: ContinuousTraining.py
───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1   │ import airflow
   2   │ from airflow.decorators import dag
   3   │ from airflow.operators.python_operator import PythonOperator
...
```