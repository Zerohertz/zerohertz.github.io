---
title: 'Home Server: External IP 변동 감지 Service'
date: 2023-08-17 15:12:50
categories:
- 0. Daily
tags:
- Home Server
---
# Introduction

Subdomain을 생성하기 위해 새 domain을 [GoDaddy](https://kr.godaddy.com/)에서 구매했다.
하지만 구매한 domain은 DDNS (Dynamic Domain Name System)가 아닌 DNS (Domain Name System)이므로 공유기에 할당된 external IP가 변경되면 새로 바꿔줘야한다.
물론 DDNS service를 연결할 수 있지만 무료로 이용하려면 제약되는 것들이 많아서 IP가 변경되는 순간 Discord Webhook로 메시지가 전달되도록 구성해보겠다.

<!-- More -->

---

# Shell Script

```bash check_ip_change.sh (Discord)
#!/bin/bash

DISCORD="${WEBHOOK_URL}"

to_discord() {
    local content=$1
    curl -H "Content-Type: application/json" \
         -X POST \
         -d "{\"content\":\"$content\"}" \
         $DISCORD
}

IP_FILE="/tmp/last_known_ip"
current_ip=$(curl -s ipecho.net/plain)
if [[ "$current_ip" != "$(cat $IP_FILE 2>/dev/null)" ]]; then
    to_discord "IP changed to $current_ip"
    echo "$current_ip" > $IP_FILE
fi
```

이 script를 실행하면 아래와 같이 잘 구동된다.

```shell
$ chmod +x check_ip_change.sh
$ ./check_ip_change.sh
```

<img src="/images/home-server-external-ip/discord.png" alt="discord" width="300" />

혹은 아래와 같이 script를 작성하면 Slack에서도 사용할 수 있다.

```bash check_ip_change.sh (Slack)
#!/bin/bash

to_slack() {
    local content=$1
    curl -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-type: application/json" \
        -d '{
                "channel": "${CHANNEL}",
                "text": "'"$content"'",
                "username": "IP",
                "icon_emoji": "${EMOJI}",
            }'
}

IP_FILE="/tmp/last_known_ip"
current_ip=$(curl -s ipecho.net/plain)
if [[ "$current_ip" != "$(cat $IP_FILE 2>/dev/null)" ]]; then
    to_slack "IP changed to $current_ip"
    echo "$current_ip" > $IP_FILE
fi
```

---

# systemd

> systemd: Linux system의 초기화 process와 service manager로, service의 시작, 중지 및 관리를 담당하며, system booting과 service 의존성 관리에 필요한 도구들을 제공

```python /etc/systemd/system/check_ip_change.service
[Unit]
Description=Check IP change and notify to Discord

[Service]
ExecStart=/path/to/check_ip_change.sh
```

```python /etc/systemd/system/check_ip_change.timer
[Unit]
Description=Run check_ip_change.service every 10 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=10min

[Install]
WantedBy=timers.target
```

```shell
$ sudo systemctl daemon-reload
$ sudo systemctl status check_ip_change.service
● check_ip_change.service - Check IP change and notify to Discord
     Loaded: loaded (/etc/systemd/system/check_ip_change.service; static; vendor preset: enabled)
     Active: inactive (dead)
$ sudo systemctl status check_ip_change.timer
● check_ip_change.timer - Run check_ip_change.service every 10 minutes
     Loaded: loaded (/etc/systemd/system/check_ip_change.timer; disabled; vendor preset: enabled)
     Active: inactive (dead)
    Trigger: n/a
   Triggers: ● check_ip_change.service

$ sudo systemctl enable check_ip_change.timer
$ sudo systemctl start check_ip_change.timer
$ sudo systemctl status check_ip_change.service
● check_ip_change.service - Check IP change and notify to Discord
     Loaded: loaded (/etc/systemd/system/check_ip_change.service; static; vendor preset: enabled)
     Active: inactive (dead) since Thu 2023-08-17 15:41:11 KST; 4s ago
TriggeredBy: ● check_ip_change.timer
    Process: 1165238 ExecStart=/home/zerohertz/sh/check_ip_change.sh (code=exited, status=0/SUCCESS)
   Main PID: 1165238 (code=exited, status=0/SUCCESS)

Aug 17 15:41:11 0hz systemd[1]: Started Check IP change and notify to Discord.
Aug 17 15:41:11 0hz systemd[1]: check_ip_change.service: Succeeded.
$ sudo systemctl status check_ip_change.timer
● check_ip_change.timer - Run check_ip_change.service every 10 minutes
     Loaded: loaded (/etc/systemd/system/check_ip_change.timer; enabled; vendor preset: enabled)
     Active: active (waiting) since Thu 2023-08-17 15:41:11 KST; 11s ago
    Trigger: Thu 2023-08-17 15:51:11 KST; 9min left
   Triggers: ● check_ip_change.service

Aug 17 15:41:11 0hz systemd[1]: Started Run check_ip_change.service every 10 minutes.
$ cat /tmp/last_known_ip
XXX.XXX.XXX.XXX
```