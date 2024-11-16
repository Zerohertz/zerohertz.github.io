---
title: Serial Communication (Arduino to Raspberry Pi)
date: 2020-08-18 13:24:03
categories:
- Etc.
tags:
- Arduino
- Raspberry Pi
- Python
---
# CMD를 통한 Serial 통신

```shell
$ sudo usermod -a -G tty pi
$ sudo usermod -a -G dialout pi
$ stty -F /dev/ttyACM0 raw 115200 cs8 clocal -cstopb
$ cat /dev/ttyACM0
```

<!-- More -->

# Python을 통한 Serial 통신

~~~Python ser.py
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
f = open('filename.csv', 'w', encoding = 'utf-8')
t = time.time()

try:
    while True:
        if ser.in_waiting != 0:
            t1 = time.time() - t
            t2 = round(t1, 5)
            t3 = str(t2)
            sensor = ser.readline()
            print(t3)
            print(sensor.decode())
            f.write(t3)
            f.write(',')
            f.write(sensor.decode())
except:
    f.close()
~~~

+ 파일명을 `serial.py`로 지정하면 오류가 날 수 있으므로 유의