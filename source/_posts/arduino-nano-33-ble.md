---
title: 'Arduino Nano 33 BLE'
date: 2020-08-12 00:08:38
categories:
- Etc.
tags:
- Arduino
- C, C++
- Raspberry Pi
- Python
---
# Arduino Nano 33 BLE

|센서 / 모듈 이름|Info|
|:--:|:--:|
|nRF52840|32Bbit, 64MHz MCU|
|NINA-B306|Bluetooth|
|LSM9DS1|9축 IMU센서|

+ RESET 버튼을 2번 눌러 Upload mode로 전환
+ `begin()` 메서드 필수

[Aruino Nano 33 BLE](https://mechasolution.com/shop/goods/goods_view.php?&goodsno=586133)

<!-- More -->

***

# Arduino IDE와 Arduino Nano 33 BLE 연결

<img width="912" alt="Board Manager" src="/images/arduino-nano-33-ble/89915513-6ab55b80-dc31-11ea-96ee-de1eac623357.png">

+ `툴` - `보드` - `보드 매니저`에서 `arduino nano 33 BLE`를 검색
+ `Arduino nRF528x Boards` 설치

<img width="1100" alt="Serial Port" src="/images/arduino-nano-33-ble/89915989-ffb85480-dc31-11ea-946e-f56a7c213746.png">

+ `툴` - `포트` - `Arduino Nano 33 BLE` 선택
+ `보드 정보 얻기`로 연결 확인 가능
+ 안된다면 케이블을 바꿔 해결 가능(데이터 전송이 가능한 케이블)

***

# IMU Sensor

<img width="730" alt="Manage Library" src="/images/arduino-nano-33-ble/89916584-be747480-dc32-11ea-98e8-2d2c457aa54f.png">

+ `툴` - `라이브러리 관리...`

<img width="912" alt="LSM9DS1" src="/images/arduino-nano-33-ble/89916551-b583a300-dc32-11ea-8d7d-549241512a9c.png">

+ 9축 IMU센서의 이름 `LSM9DS1` 검색 후 `Arduino_LSM9DS1` 라이브러리 다운로드

~~~C++ IMU.ino
#include <Arduino_LSM9DS1.h>

float acc_x, acc_y, acc_z;
float gyro_x, gyro_y, gyro_z;
float mag_x, mag_y, mag_z;

void setup() {
  Serial.begin(2000000);
  while(!Serial);
  if (!IMU.begin()) { //LSM9DSI센서 시작
    Serial.println("LSM9DSI센서 오류!");
    while (1);
  }
  Serial.println("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z");
  delay(100);
}
void loop() {
  //가속도센서
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(acc_x, acc_y, acc_z);
    Serial.print(acc_x); Serial.print(","); Serial.print(acc_y); Serial.print(","); Serial.print(acc_z); Serial.print(",");
  }
  //자이로센서
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gyro_x, gyro_y, gyro_z);
    Serial.print(gyro_x); Serial.print(","); Serial.print(gyro_y); Serial.print(","); Serial.print(gyro_z); Serial.print(",");
  }
  //지자계센서
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mag_x, mag_y, mag_z);
    Serial.print(mag_x); Serial.print(","); Serial.print(mag_y); Serial.print(","); Serial.println(mag_z);
  }
  delay(50);
}
~~~

> Graph

<img width="1204" alt="IMU" src="/images/arduino-nano-33-ble/89924286-c6d1ad00-dc3c-11ea-9da0-2ff8961262f9.png">

***

# Bluetooth Module

<img width="912" alt="스크린샷 2020-08-12 오전 1 47 07" src="/images/arduino-nano-33-ble/89924997-bcfc7980-dc3d-11ea-8533-10b7bea3d618.png">

~~~C++ PeripheralExplorer.ino
/*
  Peripheral Explorer

  This example scans for BLE peripherals until one with a particular name ("LED")
  is found. Then connects, and discovers + prints all the peripheral's attributes.

  The circuit:
  - Arduino MKR WiFi 1010, Arduino Uno WiFi Rev2 board, Arduino Nano 33 IoT,
    Arduino Nano 33 BLE, or Arduino Nano 33 BLE Sense board.

  You can use it with another board that is compatible with this library and the
  Peripherals -> LED example.

  This example code is in the public domain.
*/

#include <ArduinoBLE.h>

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");

    while (1);
  }

  Serial.println("BLE Central - Peripheral Explorer");

  // start scanning for peripherals
  BLE.scan();
}

void loop() {
  // check if a peripheral has been discovered
  BLEDevice peripheral = BLE.available();

  if (peripheral) {
    // discovered a peripheral, print out address, local name, and advertised service
    Serial.print("Found ");
    Serial.print(peripheral.address());
    Serial.print(" '");
    Serial.print(peripheral.localName());
    Serial.print("' ");
    Serial.print(peripheral.advertisedServiceUuid());
    Serial.println();

    // see if peripheral is a LED
    if (peripheral.localName() == "IMU") {
      // stop scanning
      BLE.stopScan();

      explorerPeripheral(peripheral);

      // peripheral disconnected, we are done
      while (1) {
        // do nothing
      }
    }
  }
}

void explorerPeripheral(BLEDevice peripheral) {
  // connect to the peripheral
  Serial.println("Connecting ...");

  if (peripheral.connect()) {
    Serial.println("Connected");
  } else {
    Serial.println("Failed to connect!");
    return;
  }

  // discover peripheral attributes
  Serial.println("Discovering attributes ...");
  if (peripheral.discoverAttributes()) {
    Serial.println("Attributes discovered");
  } else {
    Serial.println("Attribute discovery failed!");
    peripheral.disconnect();
    return;
  }

  // read and print device name of peripheral
  Serial.println();
  Serial.print("Device name: ");
  Serial.println(peripheral.deviceName());
  Serial.print("Appearance: 0x");
  Serial.println(peripheral.appearance(), HEX);
  Serial.println();

  // loop the services of the peripheral and explore each
  for (int i = 0; i < peripheral.serviceCount(); i++) {
    BLEService service = peripheral.service(i);

    exploreService(service);
  }

  Serial.println();

  // we are done exploring, disconnect
  Serial.println("Disconnecting ...");
  peripheral.disconnect();
  Serial.println("Disconnected");
}

void exploreService(BLEService service) {
  // print the UUID of the service
  Serial.print("Service ");
  Serial.println(service.uuid());

  // loop the characteristics of the service and explore each
  for (int i = 0; i < service.characteristicCount(); i++) {
    BLECharacteristic characteristic = service.characteristic(i);

    exploreCharacteristic(characteristic);
  }
}

void exploreCharacteristic(BLECharacteristic characteristic) {
  // print the UUID and properties of the characteristic
  Serial.print("\tCharacteristic ");
  Serial.print(characteristic.uuid());
  Serial.print(", properties 0x");
  Serial.print(characteristic.properties(), HEX);

  // check if the characteristic is readable
  if (characteristic.canRead()) {
    // read the characteristic value
    characteristic.read();

    if (characteristic.valueLength() > 0) {
      // print out the value of the characteristic
      Serial.print(", value 0x");
      printData(characteristic.value(), characteristic.valueLength());
    }
  }
  Serial.println();

  // loop the descriptors of the characteristic and explore each
  for (int i = 0; i < characteristic.descriptorCount(); i++) {
    BLEDescriptor descriptor = characteristic.descriptor(i);

    exploreDescriptor(descriptor);
  }
}

void exploreDescriptor(BLEDescriptor descriptor) {
  // print the UUID of the descriptor
  Serial.print("\t\tDescriptor ");
  Serial.print(descriptor.uuid());

  // read the descriptor value
  descriptor.read();

  // print out the value of the descriptor
  Serial.print(", value 0x");
  printData(descriptor.value(), descriptor.valueLength());

  Serial.println();
}

void printData(const unsigned char data[], int length) {
  for (int i = 0; i < length; i++) {
    unsigned char b = data[i]; 

    if (b < 16) {
      Serial.print("0");
    }

    Serial.print(b, HEX);
  }
}
~~~

~~~C++ Output
BLE Central - Peripheral Explorer
Found e8:61:57:48:1a:ef 'IMU' 1101
Connecting ...
Connected
Discovering attributes ...
Attributes discovered

Device name: IMU
Appearance: 0x0

Service 1800
	Characteristic 2a00, properties 0x2, value 0x494D55
	Characteristic 2a01, properties 0x2, value 0x0000
Service 1801
	Characteristic 2a05, properties 0x20
		Descriptor 2902, value 0x0000
Service 1101
	Characteristic 2101, properties 0x12, value 0x0A
		Descriptor 2902, value 0x0000

Disconnecting ...
Disconnected
~~~

~~~C++ peripheral.ino
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

BLEService ACC("1001");
BLEFloatCharacteristic accX("2001", BLERead | BLENotify);
BLEFloatCharacteristic accY("2002", BLERead | BLENotify);
BLEFloatCharacteristic accZ("2003", BLERead | BLENotify);

BLEService GYRO("1002");
BLEFloatCharacteristic gyroX("2011", BLERead | BLENotify);
BLEFloatCharacteristic gyroY("2012", BLERead | BLENotify);
BLEFloatCharacteristic gyroZ("2013", BLERead | BLENotify);

BLEService MAG("1003");
BLEFloatCharacteristic magX("2021", BLERead | BLENotify);
BLEFloatCharacteristic magY("2022", BLERead | BLENotify);
BLEFloatCharacteristic magZ("2023", BLERead | BLENotify);

float acc_x, acc_y, acc_z;
float gyro_x, gyro_y, gyro_z;
float mag_x, mag_y, mag_z;

void setup() {
  Serial.begin(115200);
  
  if(!BLE.begin()) {
    Serial.println("Starting BLE Failed!");
    while(1);
  }

  if (!IMU.begin()) { //LSM9DSI센서 시작
    Serial.println("LSM9DSI센서 오류!");
    while (1);
  }
  
  BLE.setDeviceName("IMU");
  BLE.setLocalName("IMU");
  
  BLE.setAdvertisedService(ACC);
  BLE.setAdvertisedService(GYRO);
  BLE.setAdvertisedService(MAG);
  ACC.addCharacteristic(accX);
  ACC.addCharacteristic(accY);
  ACC.addCharacteristic(accZ);
  GYRO.addCharacteristic(gyroX);
  GYRO.addCharacteristic(gyroY);
  GYRO.addCharacteristic(gyroZ);
  MAG.addCharacteristic(magX);
  MAG.addCharacteristic(magY);
  MAG.addCharacteristic(magZ);
  BLE.addService(ACC);
  BLE.addService(GYRO);
  BLE.addService(MAG);
  BLE.setConnectable(true);
  BLE.setAdvertisingInterval(100);
  BLE.advertise();
  Serial.println("Bluetooth Device Active, Waiting for Connections...");
}

void loop() {
  BLEDevice central = BLE.central();

  if(central) {
    Serial.print("Connected to Central: ");
    Serial.println(central.address());
    while(central.connected()) {
      IMU.readAcceleration(acc_x, acc_y, acc_z);
      IMU.readGyroscope(gyro_x, gyro_y, gyro_z);
      IMU.readMagneticField(mag_x, mag_y, mag_z);      
      accX.writeValue(acc_x);
      accY.writeValue(acc_y);
      accZ.writeValue(acc_z);
      gyroX.writeValue(gyro_x);
      gyroY.writeValue(gyro_y);
      gyroZ.writeValue(gyro_z);
      magX.writeValue(mag_x);
      magY.writeValue(mag_y);
      magZ.writeValue(mag_z);
      Serial.println(acc_x);
    }
  }
  Serial.print("Disconnected from Central: ");
  Serial.println(BLE.address());
}
~~~

~~~C++ central.ino
#include <ArduinoBLE.h>

union dat{
  unsigned char asdf[4];
  float zxcv;
};

float getData(const unsigned char data[], int length) {
  dat dat;
  for (int i = 0; i < length; i++) {
    dat.asdf[i] = data[i]; 
    }
  return dat.zxcv;
}

void printcsv(BLECharacteristic c1, BLECharacteristic c2, BLECharacteristic c3, BLECharacteristic c4, BLECharacteristic c5, BLECharacteristic c6, BLECharacteristic c7, BLECharacteristic c8, BLECharacteristic c9){
  c1.read();
  c2.read();
  c3.read();
  c4.read();
  c5.read();
  c6.read();
  c7.read();
  c8.read();
  c9.read(); 
  float f1=getData(c1.value(), c1.valueLength());
  float f2=getData(c2.value(), c2.valueLength());
  float f3=getData(c3.value(), c3.valueLength());
  float f4=getData(c4.value(), c4.valueLength());
  float f5=getData(c5.value(), c5.valueLength());
  float f6=getData(c6.value(), c6.valueLength());
  float f7=getData(c7.value(), c7.valueLength());
  float f8=getData(c8.value(), c8.valueLength());
  float f9=getData(c9.value(), c9.valueLength());
  Serial.print(f1);
  Serial.print(',');
  Serial.print(f2);
  Serial.print(',');
  Serial.print(f3);
  Serial.print(',');
  Serial.print(f4);
  Serial.print(',');
  Serial.print(f5);
  Serial.print(',');
  Serial.print(f6);
  Serial.print(',');
  Serial.print(f7);
  Serial.print(',');
  Serial.print(f8);
  Serial.print(',');
  Serial.print(f9);
  Serial.print('\n');
}

void setup() {
  Serial.begin(115200);

  if(!BLE.begin()) {
    Serial.println("Starting BLE Failed!");
    while(1);
  }
  BLE.scan();
}

void loop() {
  BLEDevice peripheral = BLE.available();

  if(peripheral){
    if(peripheral.localName()=="IMU"){
      BLE.stopScan();
      if(peripheral.connect()){
        Serial.println("Connect1");
      }
      else{
        return;
      }
      if(peripheral.discoverAttributes()){
        Serial.println("Connect2");
      }
      else{
        return;
      }
      BLEService acc=peripheral.service("1001");
      BLECharacteristic accx=acc.characteristic("2001");
      BLECharacteristic accy=acc.characteristic("2002");
      BLECharacteristic accz=acc.characteristic("2003");
      BLEService gyro=peripheral.service("1002");
      BLECharacteristic gyrox=gyro.characteristic("2011");
      BLECharacteristic gyroy=gyro.characteristic("2012");
      BLECharacteristic gyroz=gyro.characteristic("2013");
      BLEService mag=peripheral.service("1003");
      BLECharacteristic magx=mag.characteristic("2021");
      BLECharacteristic magy=mag.characteristic("2022");
      BLECharacteristic magz=mag.characteristic("2023");
      while(true){
//        accx.read();
//        float f1=getData(accx.value(),accx.valueLength());
//        Serial.print(f1);
//        Serial.print(',');
//        accy.read();
//        float f2=getData(accy.value(),accy.valueLength());
//        Serial.print(f2);
//        Serial.print(',');
//        accz.read();
//        float f3=getData(accz.value(),accz.valueLength());
//        Serial.println(f3);
        if(peripheral.connected()){
          printcsv(accx,accy,accz,gyrox,gyroy,gyroz,magx,magy,magz);
        }
        else{
          peripheral.disconnect();
          return;
        }
      }
    }
  }
  BLE.scan();
  Serial.println("rescan");
}
~~~