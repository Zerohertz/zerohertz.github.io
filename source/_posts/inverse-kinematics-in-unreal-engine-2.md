---
title: Inverse Kinematics in Unreal Engine (2)
date: 2019-06-02 15:55:01
categories:
- Etc.
tags:
- C, C++
- Unreal Engine
---
# Unreal engine과 Inverse Kinematics의 차원이 달랐다

> IKBPLibrary.cpp

<!-- more -->
~~~C++
// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "IKBPLibrary.h"
#include "IK.h"

UIKBPLibrary::UIKBPLibrary(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{

}

float UIKBPLibrary::IKSampleFunction(float Param)
{
	return -1;
}

void UIKBPLibrary::InverseKinematics(FVector pos, FRotator rot, float p, float &Joint0, float &Joint1, float &Joint2, float &Joint3, float &Joint4, float &Joint5, float &Joint6)
{
	FILE* fp = fopen("test.txt", "w");
	float JointAngle[NB_JOINT] = {};
	float Alpha[NB_JOINT] = {};
	float A[NB_JOINT] = {};
	float D[NB_JOINT] = {};

	Alpha[0] = -90 * PI / 180.0;
	Alpha[1] = 90 * PI / 180.0;
	Alpha[2] = -90 * PI / 180.0;
	Alpha[3] = 90 * PI / 180.0;
	Alpha[4] = -90 * PI / 180.0;
	Alpha[5] = 90 * PI / 180.0;
	Alpha[6] = 0;

	A[0] = A[1] = A[2] = A[3] = A[4] = A[5] = A[6] = 0;

	D[0] = 360;
	D[1] = 0;
	D[2] = 420;
	D[3] = 0;
	D[4] = 400;
	D[5] = 0;
	D[6] = 110;

	/*int test;
	test = 10;
	
	pos.X = test;
	pos.Y = test;
	pos.Z = test;

	rot.Roll = test;
	rot.Pitch = test;
	rot.Yaw = test;*/

	FVector err;
	err = rot.RotateVector(FVector(0, 0, D[6]));
	err.X = -err.X;
	err.Y = -err.Y;

	FVector wristPos = pos - err;

	FVector WristPos;
	WristPos = wristPos;

	fprintf(fp, "wristPos : %f\t, %f\t, %f\t", wristPos.X, wristPos.Y, wristPos.Z);
	FVector shoulderToWrist = wristPos - FVector(0, 0, D[0]);

	FVector unitStoW = shoulderToWrist / shoulderToWrist.Size();

	FRotator k;
	FRotationMatrix kw(k);
	kw.M[0][0] = 0; kw.M[0][1] = -unitStoW.Z; kw.M[0][2] = unitStoW.Y;
	kw.M[1][0] = unitStoW.Z; kw.M[1][1] = 0; kw.M[1][2] = -unitStoW.X;
	kw.M[2][0] = -unitStoW.Y; kw.M[2][1] = unitStoW.X; kw.M[2][2] = 0;

	FRotator identity;
	FRotationMatrix Identity(identity);
	Identity.M[0][0] = 1; Identity.M[0][1] = 0; Identity.M[0][2] = 0;
	Identity.M[1][0] = 0; Identity.M[1][1] = 1; Identity.M[1][2] = 0;
	Identity.M[2][0] = 0; Identity.M[2][1] = 0; Identity.M[2][2] = 1;

	float theta4Prime = D[2] * D[2] + D[4] * D[4] - shoulderToWrist.SizeSquared();
	theta4Prime = theta4Prime / (2.0 * D[2] * D[4]);
	JointAngle[3] = PI - acos(theta4Prime);

	float theta1Prime = atan2(wristPos.Y, wristPos.X);
	float alpha = (wristPos.Z - D[0]) / shoulderToWrist.Size();
	alpha = asin(alpha);
	float beta = D[2] * D[2] + shoulderToWrist.SizeSquared() - D[4] * D[4];
	beta = beta / (2 * D[2] * shoulderToWrist.Size());
	beta = acos(beta);
	float theta2Prime = PI / 2.0 - alpha - beta;

	FRotator Prime;
	FRotationMatrix rotPrime(Prime);
	rotPrime.M[0][0] = cos(theta1Prime)*cos(theta2Prime); rotPrime.M[0][1] = -cos(theta1Prime)*sin(theta2Prime); rotPrime.M[0][2] = -sin(theta1Prime);
	rotPrime.M[1][0] = cos(theta2Prime)*sin(theta1Prime); rotPrime.M[1][1] = -sin(theta1Prime)*sin(theta2Prime); rotPrime.M[1][2] = cos(theta1Prime);
	rotPrime.M[2][0] = -sin(theta2Prime); rotPrime.M[2][1] = -cos(theta2Prime); rotPrime.M[2][2] = 0;

	JointAngle[0] = atan2(wristPos.Y, wristPos.X);
	JointAngle[1] = theta2Prime;
	JointAngle[2] = 0;
	//////////////////////////////////////////////
	FMatrix T;
	T.M[0][0] = 1; T.M[0][1] = 0; T.M[0][2] = 0; T.M[0][3] = 0; //T.identity()
	T.M[1][0] = 0; T.M[1][1] = 1; T.M[1][2] = 0; T.M[1][3] = 0;
	T.M[2][0] = 0; T.M[2][1] = 0; T.M[2][2] = 1; T.M[2][3] = 0;
	T.M[3][0] = 0; T.M[3][1] = 0; T.M[3][2] = 0; T.M[3][3] = 1;

	for (int i = 0; i < 4; i++) {
		FMatrix transformationMatirx;
		transformationMatirx.M[0][0] = cos(JointAngle[i]); transformationMatirx.M[0][1] = -sin(JointAngle[i]) * cos(Alpha[i]); transformationMatirx.M[0][2] = sin(JointAngle[i])*sin(Alpha[i]); transformationMatirx.M[0][3] = A[i] * cos(JointAngle[i]);
		transformationMatirx.M[1][0] = sin(JointAngle[i]); transformationMatirx.M[1][1] = cos(JointAngle[i]) * cos(Alpha[i]); transformationMatirx.M[1][2] = -cos(JointAngle[i])*sin(Alpha[i]); transformationMatirx.M[1][3] = A[i] * sin(JointAngle[i]);
		transformationMatirx.M[2][0] = 0; transformationMatirx.M[2][1] = sin(Alpha[i]); transformationMatirx.M[2][2] = cos(Alpha[i]); transformationMatirx.M[2][3] = D[i];
		transformationMatirx.M[3][0] = 0; transformationMatirx.M[3][1] = 0; transformationMatirx.M[3][2] = 0; transformationMatirx.M[3][3] = 1;
		FMatrix ti = transformationMatirx;
		T = T * ti;
	}
	fprintf(fp, "\n T = \n");

	for (int i = 0; i < 4; i++) {
		for(int j=0; j<4; j++){
			fprintf(fp, "%.2f\t", T.M[i][j]);
		}
		fprintf(fp, "\n");
	}
	
	FMatrix rot041;
	FMatrix Rot04;

	float j[2] = {};

	rot041 = Rot04 = T;
	for (int i = 0; i < 3; i++) {
		rot041.M[i][3] = 0;
		Rot04.M[i][3] = 0;
		j[i] = T.M[i][3];
	}

	fprintf(fp, "\n Rot04 = \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", Rot04.M[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n rot041 = \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", rot041.M[i][j]);
		}
		fprintf(fp, "\n");
	}


	FVector Pos04;
	Pos04.X = j[0];
	Pos04.Y = j[1];
	Pos04.Z = j[2];
	
	FMatrix rot47;
	rot47.M[0][0] = 1; rot47.M[0][1] = 0; rot47.M[0][2] = 0; rot47.M[0][3] = 0;
	rot47.M[1][0] = 0; rot47.M[1][1] = 1; rot47.M[1][2] = 0; rot47.M[1][3] = 0;
	rot47.M[2][0] = 0; rot47.M[2][1] = 0; rot47.M[2][2] = 1; rot47.M[2][3] = 0;
	rot47.M[3][0] = 0; rot47.M[3][1] = 0; rot47.M[3][2] = 0; rot47.M[3][3] = 1;

	FRotationMatrix ROT1(rot);
	FMatrix ROT = ROT1.GetTransposed();
	ROT.M[0][2] = -ROT.M[0][2];
	ROT.M[1][2] = -ROT.M[1][2];
	ROT.M[2][0] = -ROT.M[2][0];
	ROT.M[2][1] = -ROT.M[2][1];

	fprintf(fp, "\n rot= \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", ROT.M[i][j]);
		}
		fprintf(fp, "\n");
	}

	rot47 = rot041.GetTransposed() * ROT;

	fprintf(fp, "\n rot47 = \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", rot47.M[i][j]);
		}
		fprintf(fp, "\n");
	}


	if (fabs(rot47.M[0][2]) > 0.000001)
	{
		if (rot47.M[0][2] > 0)
			JointAngle[4] = atan(rot47.M[1][2] / rot47.M[0][2]);
		else
			JointAngle[4] = atan(rot47.M[1][2] / rot47.M[0][2]) + PI;
	}
	JointAngle[5] = acos(rot47.M[2][2]);

	if (fabs(rot47.M[2][0]) > 0.000001)
	{
		if (rot47.M[2][0] > 0)
			JointAngle[6] = atan(-rot47.M[2][1] / rot47.M[2][0]) + PI;
		else
			JointAngle[6] = atan(rot47.M[2][1] / rot47.M[2][0]);
	}
	
	Joint0 = JointAngle[0];
	Joint1 = JointAngle[1];
	Joint2 = JointAngle[2];
	Joint3 = JointAngle[3];
	Joint4 = JointAngle[4];
	Joint5 = JointAngle[5];
	Joint6 = JointAngle[6];
	fclose(fp);
}
~~~
***
# 변수 직관성, fprintf 수정

> IKBPLibrary.cpp

~~~C++
// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "IKBPLibrary.h"
#include "IK.h"

UIKBPLibrary::UIKBPLibrary(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{

}

float UIKBPLibrary::IKSampleFunction(float Param)
{
	return -1;
}

void UIKBPLibrary::InverseKinematics(FVector pos, FRotator rot, float p, float &Joint0, float &Joint1, float &Joint2, float &Joint3, float &Joint4, float &Joint5, float &Joint6)
{
	//FILE* fp = fopen("test.txt", "w");
	float JointAngle[NB_JOINT] = {};
	float Alpha[NB_JOINT] = {};
	float A[NB_JOINT] = {};
	float D[NB_JOINT] = {};

	Alpha[0] = -90 * PI / 180.0;
	Alpha[1] = 90 * PI / 180.0;
	Alpha[2] = -90 * PI / 180.0;
	Alpha[3] = 90 * PI / 180.0;
	Alpha[4] = -90 * PI / 180.0;
	Alpha[5] = 90 * PI / 180.0;
	Alpha[6] = 0;

	A[0] = A[1] = A[2] = A[3] = A[4] = A[5] = A[6] = 0;

	D[0] = 360;
	D[1] = 0;
	D[2] = 420;
	D[3] = 0;
	D[4] = 400;
	D[5] = 0;
	D[6] = 110;

	/*int test;
	test = 10;
	
	pos.X = test;
	pos.Y = test;
	pos.Z = test;

	rot.Roll = test;
	rot.Pitch = test;
	rot.Yaw = test;*/

	FVector rotmulvec;
	rotmulvec = rot.RotateVector(FVector(0, 0, D[6]));
	rotmulvec.X = -rotmulvec.X;
	rotmulvec.Y = -rotmulvec.Y;

	FVector wristPos = pos - rotmulvec;

	FVector WristPos;
	WristPos = wristPos;

	//fprintf(fp, "wristPos : %f\t, %f\t, %f\t", wristPos.X, wristPos.Y, wristPos.Z);
	FVector shoulderToWrist = wristPos - FVector(0, 0, D[0]);

	FVector unitStoW = shoulderToWrist / shoulderToWrist.Size();

	FRotator k;
	FRotationMatrix kw(k);
	kw.M[0][0] = 0; kw.M[0][1] = -unitStoW.Z; kw.M[0][2] = unitStoW.Y;
	kw.M[1][0] = unitStoW.Z; kw.M[1][1] = 0; kw.M[1][2] = -unitStoW.X;
	kw.M[2][0] = -unitStoW.Y; kw.M[2][1] = unitStoW.X; kw.M[2][2] = 0;

	FRotator identity;
	FRotationMatrix Identity(identity);
	Identity.M[0][0] = 1; Identity.M[0][1] = 0; Identity.M[0][2] = 0;
	Identity.M[1][0] = 0; Identity.M[1][1] = 1; Identity.M[1][2] = 0;
	Identity.M[2][0] = 0; Identity.M[2][1] = 0; Identity.M[2][2] = 1;

	float theta4Prime = D[2] * D[2] + D[4] * D[4] - shoulderToWrist.SizeSquared();
	theta4Prime = theta4Prime / (2.0 * D[2] * D[4]);
	JointAngle[3] = PI - acos(theta4Prime);

	float theta1Prime = atan2(wristPos.Y, wristPos.X);
	float alpha = (wristPos.Z - D[0]) / shoulderToWrist.Size();
	alpha = asin(alpha);
	float beta = D[2] * D[2] + shoulderToWrist.SizeSquared() - D[4] * D[4];
	beta = beta / (2 * D[2] * shoulderToWrist.Size());
	beta = acos(beta);
	float theta2Prime = PI / 2.0 - alpha - beta;

	FRotator Prime;
	FRotationMatrix rotPrime(Prime);
	rotPrime.M[0][0] = cos(theta1Prime)*cos(theta2Prime); rotPrime.M[0][1] = -cos(theta1Prime)*sin(theta2Prime); rotPrime.M[0][2] = -sin(theta1Prime);
	rotPrime.M[1][0] = cos(theta2Prime)*sin(theta1Prime); rotPrime.M[1][1] = -sin(theta1Prime)*sin(theta2Prime); rotPrime.M[1][2] = cos(theta1Prime);
	rotPrime.M[2][0] = -sin(theta2Prime); rotPrime.M[2][1] = -cos(theta2Prime); rotPrime.M[2][2] = 0;

	JointAngle[0] = atan2(wristPos.Y, wristPos.X);
	JointAngle[1] = theta2Prime;
	JointAngle[2] = 0;
	
	FMatrix T;
	T.M[0][0] = 1; T.M[0][1] = 0; T.M[0][2] = 0; T.M[0][3] = 0; //T.identity()
	T.M[1][0] = 0; T.M[1][1] = 1; T.M[1][2] = 0; T.M[1][3] = 0;
	T.M[2][0] = 0; T.M[2][1] = 0; T.M[2][2] = 1; T.M[2][3] = 0;
	T.M[3][0] = 0; T.M[3][1] = 0; T.M[3][2] = 0; T.M[3][3] = 1;

	for (int i = 0; i < 4; i++) {
		FMatrix transformationMatirx;
		transformationMatirx.M[0][0] = cos(JointAngle[i]); transformationMatirx.M[0][1] = -sin(JointAngle[i]) * cos(Alpha[i]); transformationMatirx.M[0][2] = sin(JointAngle[i])*sin(Alpha[i]); transformationMatirx.M[0][3] = A[i] * cos(JointAngle[i]);
		transformationMatirx.M[1][0] = sin(JointAngle[i]); transformationMatirx.M[1][1] = cos(JointAngle[i]) * cos(Alpha[i]); transformationMatirx.M[1][2] = -cos(JointAngle[i])*sin(Alpha[i]); transformationMatirx.M[1][3] = A[i] * sin(JointAngle[i]);
		transformationMatirx.M[2][0] = 0; transformationMatirx.M[2][1] = sin(Alpha[i]); transformationMatirx.M[2][2] = cos(Alpha[i]); transformationMatirx.M[2][3] = D[i];
		transformationMatirx.M[3][0] = 0; transformationMatirx.M[3][1] = 0; transformationMatirx.M[3][2] = 0; transformationMatirx.M[3][3] = 1;
		FMatrix ti = transformationMatirx;
		T = T * ti;
	}
	/*fprintf(fp, "\n T = \n");

	for (int i = 0; i < 4; i++) {
		for(int j=0; j<4; j++){
			fprintf(fp, "%.2f\t", T.M[i][j]);
		}
		fprintf(fp, "\n");
	}*/
	
	FMatrix rot04;
	FMatrix Rot04;
	FVector Pos04;

	rot04 = Rot04 = T;

	for (int i = 0; i < 3; i++) {
		rot04.M[i][3] = 0;
		Rot04.M[i][3] = 0;
		Pos04[i] = T.M[i][3];
	}

	/*fprintf(fp, "\n Rot04 = \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", Rot04.M[i][j]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n rot04 = \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", rot04.M[i][j]);
		}
		fprintf(fp, "\n");
	}*/

	
	FMatrix rot47;
	rot47.M[0][0] = 1; rot47.M[0][1] = 0; rot47.M[0][2] = 0; rot47.M[0][3] = 0;
	rot47.M[1][0] = 0; rot47.M[1][1] = 1; rot47.M[1][2] = 0; rot47.M[1][3] = 0;
	rot47.M[2][0] = 0; rot47.M[2][1] = 0; rot47.M[2][2] = 1; rot47.M[2][3] = 0;
	rot47.M[3][0] = 0; rot47.M[3][1] = 0; rot47.M[3][2] = 0; rot47.M[3][3] = 1;

	FRotationMatrix ROT1(rot);
	FMatrix ROT = ROT1.GetTransposed();
	ROT.M[0][2] = -ROT.M[0][2];
	ROT.M[1][2] = -ROT.M[1][2];
	ROT.M[2][0] = -ROT.M[2][0];
	ROT.M[2][1] = -ROT.M[2][1];

	/*fprintf(fp, "\n rot= \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", ROT.M[i][j]);
		}
		fprintf(fp, "\n");
	}*/

	rot47 = rot04.GetTransposed() * ROT;

	/*fprintf(fp, "\n rot47 = \n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			fprintf(fp, "%.2f\t", rot47.M[i][j]);
		}
		fprintf(fp, "\n");
	}*/


	if (fabs(rot47.M[0][2]) > 0.000001)
	{
		if (rot47.M[0][2] > 0)
			JointAngle[4] = atan(rot47.M[1][2] / rot47.M[0][2]);
		else
			JointAngle[4] = atan(rot47.M[1][2] / rot47.M[0][2]) + PI;
	}
	JointAngle[5] = acos(rot47.M[2][2]);

	if (fabs(rot47.M[2][0]) > 0.000001)
	{
		if (rot47.M[2][0] > 0)
			JointAngle[6] = atan(-rot47.M[2][1] / rot47.M[2][0]) + PI;
		else
			JointAngle[6] = atan(rot47.M[2][1] / rot47.M[2][0]);
	}
	
	Joint0 = JointAngle[0];
	Joint1 = JointAngle[1];
	Joint2 = JointAngle[2];
	Joint3 = JointAngle[3];
	Joint4 = JointAngle[4];
	Joint5 = JointAngle[5];
	Joint6 = JointAngle[6];
	//fclose(fp);
}
~~~
![Joint0](https://user-images.githubusercontent.com/42334717/60789463-220fb100-a19a-11e9-8743-e8543fbe595e.png)
![Joint1](https://user-images.githubusercontent.com/42334717/60789465-220fb100-a19a-11e9-9d8c-57d47a671b75.png)
![Joint2](https://user-images.githubusercontent.com/42334717/60789466-220fb100-a19a-11e9-95b4-11bd2fba7d9c.png)
![Joint3](https://user-images.githubusercontent.com/42334717/60789468-220fb100-a19a-11e9-9d64-82073fd75452.png)
![Joint4](https://user-images.githubusercontent.com/42334717/60789470-22a84780-a19a-11e9-92f0-19699fdd1075.png)
![Joint5](https://user-images.githubusercontent.com/42334717/60789471-22a84780-a19a-11e9-8a79-1fdeee85bd42.png)
![Joint6](https://user-images.githubusercontent.com/42334717/60789472-22a84780-a19a-11e9-83a4-8c9b6a61bb7a.png)
![원래 소스와 비교](https://user-images.githubusercontent.com/42334717/60789473-22a84780-a19a-11e9-92e1-3c420f2c861c.png)

+ 값이 모두 일치함을 볼 수 있다
+ `Joint`값을 각각의 `Joint`에 각도값을 입력하는 블루프린트를 구현