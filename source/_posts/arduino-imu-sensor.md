---
title: IMU Sensor
date: 2019-05-02 08:04:32
categories:
- Etc.
tags:
- Arduino
- C, C++
- B.S. Course Work
---
# 아두이노와 C++의 연결

> IMU_SensorView.cpp

~~~C++
void CIMUSensorView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
	LEFT_DOWN = false;
	RIGHT_DOWN = false;
	SetTimer(1, 10, NULL);
	InitGL();

	m_IMU9250.readObjData("../data/9250-1.txt");
	if (m_Serial.connect("COM3"))
		printf("Serial port is connected!!\n");
	else
		printf("Sercial connection fail!!\n");
}
~~~
<!-- more -->
***
# ConvertString2Float()

+ 문자열을 슬라이싱한 후에 `atof()`를 사용하여 전환해준다

> IMU_SensorView.cpp

~~~C++
void CIMUSensorView::convert2Float(char* data, float& roll, float& pitch)
{
	char data1[64] = {};
	char data2[64] = {};
	int flag = 0;
	int a = 0;
	int b = 0;
	while (1)
	{
		if (flag == 2)
			break;

		if (flag == 1)
		{
			data2[b] = data[a];
			if (data[a] == '\n')
			{
				flag = 2;
			}
			b++;
		}

		if (flag == 0)
		{
			data1[a] = data[a];
			if (data[a] == '\t')
			{
				flag = 1;
			}
		}
		a++;
	}
	float d1 = atof(data1);
	float d2 = atof(data2);

	roll = -d1 * PI / 180.0;
	pitch = d2 * PI / 180.0;
}
~~~

# ComputeRotationMatrix()

> IMU_SensorView.cpp

~~~C++
Mat3x3f CIMUSensorView::computeRotationMatrix(float roll, float pitch)
{
	Mat3x3f rot1, rot2;
	computeRotationMatrix(Vec3f(1, 0, 0), roll, rot1);
	computeRotationMatrix(Vec3f(0, 1, 0), pitch, rot2);
	return rot2 * rot1;
}

void CIMUSensorView::computeRotationMatrix(Vec3f axis, float angle, Mat3x3f& rot)
{
	float x, y, z;
	x = axis[0];
	y = axis[1];
	z = axis[2];

	rot(0, 0) = x * x + (y * y + z * z) * cos(angle);
	rot(1, 1) = y * y + (x * x + z * z) * cos(angle);
	rot(2, 2) = z * z + (x * x + y * y) * cos(angle);
	rot(0, 1) = (1 - cos(angle)) * x * y + z * sin(angle);
	rot(1, 0) = (1 - cos(angle)) * x * y - z * sin(angle);
	rot(0, 2) = (1 - cos(angle)) * x * z - y * sin(angle);
	rot(2, 0) = (1 - cos(angle)) * z * x + y * sin(angle);
	rot(1, 2) = (1 - cos(angle)) * y * z + x * sin(angle);
	rot(2, 1) = (1 - cos(angle)) * z * y - x * sin(angle);

	rot.transpose();
};
~~~