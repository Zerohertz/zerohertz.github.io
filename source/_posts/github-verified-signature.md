---
title: 'GitHub: Verified Signature'
date: 2023-08-09 10:53:58
categories:
- Etc.
tags:
- GitHub
---
# Introduction

> GitHub에 commit을 push하고 log에 `Verified`를 남기는 방법!

![verfied](/images/github-verified-signature/verfied.png)

<!-- More -->

---
# GPG Key 생성

```shell
$ gpg --full-generate-key
gpg (GnuPG) 2.2.27; Copyright (C) 2021 Free Software Foundation, Inc.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

gpg: directory '/home/zerohertz/.gnupg' created
gpg: keybox '/home/zerohertz/.gnupg/pubring.kbx' created
Please select what kind of key you want:
   (1) RSA and RSA (default)
   (2) DSA and Elgamal
   (3) DSA (sign only)
   (4) RSA (sign only)
  (14) Existing key from card
Your selection? 1
RSA keys may be between 1024 and 4096 bits long.
What keysize do you want? (3072) 4096
Requested keysize is 4096 bits
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 0
Key does not expire at all
Is this correct? (y/N) y

GnuPG needs to construct a user ID to identify your key.

Real name: Zerohertz
Email address: ohg3417@gmail.com
Comment: 
You selected this USER-ID:
    "Zerohertz <ohg3417@gmail.com>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? O
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
gpg: /home/zerohertz/.gnupg/trustdb.gpg: trustdb created
gpg: key ${Pub_Key} marked as ultimately trusted
gpg: directory '/home/zerohertz/.gnupg/openpgp-revocs.d' created
gpg: revocation certificate stored as '/home/zerohertz/.gnupg/openpgp-revocs.d/${Fingerprint}.rev'
public and secret key created and signed.

pub   rsa4096 2023-08-09 [SC]
      ${Fingerprint}
uid                      Zerohertz <ohg3417@gmail.com>
sub   rsa4096 2023-08-09 [E]
$ gpg --armor --export ${Fingerprint}
-----BEGIN PGP PUBLIC KEY BLOCK-----

mQINBGTS2LoBEACxnsidkifzeZDzGZseKanqKPYATUr67v+8ELV98NUzR/bTkDKY
...
=Ncj3
-----END PGP PUBLIC KEY BLOCK-----
```

---

# GPG key 등록

맨 마지막 명령어에서 출력된 값들을 아래의 화면 처럼 `New GPG key`를 누르고 추가하면 된다.

![gpg](/images/github-verified-signature/gpg.png)

---

# 환경 Setup

Git을 사용할 환경에서 `gpg-agent`를 설정한다.

```C ~/.gnupg/gpg-agent.conf
default-cache-ttl 28800
max-cache-ttl 28800
```

```shell
$ killall gpg-agent
```

그리고 자신이 사용하는 shell 설정 파일에 아래 코드를 추가한다. ([ZSH를 사용할 경우 `export GPG_TTY=$TTY`를 사용](https://github.com/romkatv/powerlevel10k/blob/master/README.md#how-do-i-export-gpg_tty-when-using-instant-prompt))

```bash ~/.zshrc
# export GPG_TTY=$(tty)
export GPG_TTY=$TTY
```

마지막으로 Git을 설정한다.

```shell
$ git config --global user.signingkey ${Fingerprint}
$ git config --global commit.gpgsign true
$ git config --global gpg.program gpg
```

그럼 끝 !

---

# Etc.

> GPG Key 확인 및 삭제

```shell
$ gpg --list-keys
/home/zerohertz/.gnupg/pubring.kbx
----------------------------------
pub   rsa4096 2023-08-09 [SC]
      ${Fingerprint}
uid           [ultimate] Zerohertz <ohg3417@gmail.com>
sub   rsa4096 2023-08-09 [E]
$ gpg --list-secret-keys
/home/zerohertz/.gnupg/pubring.kbx
----------------------------------
sec   rsa4096 2023-08-09 [SC]
      ${Fingerprint}
uid           [ultimate] Zerohertz <ohg3417@gmail.com>
ssb   rsa4096 2023-08-09 [E]
$ gpg --delete-secret-key ${Fingerprint}
gpg (GnuPG) 2.2.27; Copyright (C) 2021 Free Software Foundation, Inc.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.


sec  rsa4096/${Pub_Key} 2023-08-09 Zerohertz <ohg3417@gmail.com>

Delete this key from the keyring? (y/N) y
This is a secret key! - really delete? (y/N) y
gpg --delete-keys Zerohertz
gpg (GnuPG) 2.2.27; Copyright (C) 2021 Free Software Foundation, Inc.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.


pub  rsa4096/${Pub_Key} 2023-08-09 Zerohertz <ohg3417@gmail.com>

Delete this key from the keyring? (y/N) y
```