---
title: Inverse Kinematics in Unreal Engine (1)
date: 2019-05-29 15:54:57
categories:
- Etc.
tags:
- C, C++
- Unreal Engine
---
# Inverse Kinematics를 Unreal engine에 구현

> IKBPLibrary.h

<!-- more -->
~~~C++
// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "IKBPLibrary.generated.h"

#define NB_LINK 8
#define NB_JOINT 7
#define PI 3.141592

/* 
*	Function library class.
*	Each function in it is expected to be static and represents blueprint node that can be called in any blueprint.
*
*	When declaring function you can define metadata for the node. Key function specifiers will be BlueprintPure and BlueprintCallable.
*	BlueprintPure - means the function does not affect the owning object in any way and thus creates a node without Exec pins.
*	BlueprintCallable - makes a function which can be executed in Blueprints - Thus it has Exec pins.
*	DisplayName - full name of the node, shown when you mouse over the node and in the blueprint drop down menu.
*				Its lets you name the node using characters not allowed in C++ function names.
*	CompactNodeTitle - the word(s) that appear on the node.
*	Keywords -	the list of keywords that helps you to find node when you search for it using Blueprint drop-down menu. 
*				Good example is "Print String" node which you can find also by using keyword "log".
*	Category -	the category your node will be under in the Blueprint drop-down menu.
*
*	For more info on custom blueprint nodes visit documentation:
*	https://wiki.unrealengine.com/Custom_Blueprint_Node_Creation
*/
UCLASS()
class UIKBPLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_UCLASS_BODY()

public:
	UIKBPLibrary();

	UFUNCTION(BlueprintCallable, meta = (DisplayName = "Sucess", Keywords = "IK"), Category = "IK")
	static float IKSampleFunction(float Param);

	UFUNCTION(BlueprintCallable, meta = (DisplayName = "InverseKinematics", Keywords = "IK"), Category = "IK")
	static void InverseKinematics(FVector pos, FRotator rot, float &Joint0, float &Joint1, float &Joint2, float &Joint3, float &Joint4, float &Joint5, float &Joint6);
};
~~~

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

void UIKBPLibrary::InverseKinematics(FVector pos, FRotator rot, float &Joint0, float &Joint1, float &Joint2, float &Joint3, float &Joint4, float &Joint5, float &Joint6)
{
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

	FVector wristPos = pos - rot.RotateVector(FVector(0, 0, D[6]));

	FVector WristPos;
	WristPos = wristPos;

	FVector shoulderToWrist = wristPos - FVector(0, 0, D[0]);

	FVector unitStoW = shoulderToWrist / shoulderToWrist.Size();

	float psi = 0;

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

	FRotator rot04;
	FMatrix T;
	T.M[0][0] = 1; T.M[0][1] = 0; T.M[0][2] = 0; T.M[0][3] = 0; //T.identity()
	T.M[1][0] = 0; T.M[1][1] = 1; T.M[1][2] = 0; T.M[1][3] = 0;
	T.M[2][0] = 0; T.M[2][1] = 0; T.M[2][2] = 1; T.M[2][3] = 0;
	T.M[3][0] = 0; T.M[3][1] = 0; T.M[3][2] = 0; T.M[3][3] = 1;

	for (int i = 0; i < 4; i++) {
		FMatrix transformationMatirx;
		transformationMatirx.M[0][0] = cos(JointAngle[i]); transformationMatirx.M[0][1] = -sin(JointAngle[i])*cos(Alpha[i]); transformationMatirx.M[0][2] = sin(JointAngle[i])*cos(Alpha[i]); transformationMatirx.M[0][3] = A[i] * cos(JointAngle[i]);
		transformationMatirx.M[1][0] = sin(JointAngle[i]); transformationMatirx.M[1][1] = 1; transformationMatirx.M[1][2] = -cos(JointAngle[i])*sin(Alpha[i]); transformationMatirx.M[1][3] = A[i] * sin(JointAngle[i]);
		transformationMatirx.M[2][0] = 0; transformationMatirx.M[2][1] = sin(Alpha[i]); transformationMatirx.M[2][2] = cos(Alpha[i]); transformationMatirx.M[2][3] = D[i];
		transformationMatirx.M[3][0] = 0; transformationMatirx.M[3][1] = 0; transformationMatirx.M[3][2] = 0; transformationMatirx.M[3][3] = 1;
		T = T * transformationMatirx;
	}

	FRotator Roto4;
	FRotationMatrix Rot04(Roto4);
	FRotationMatrix rot041(rot04);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3, j++;) {
			Rot04.M[i][j] = rot041.M[i][j] = T.M[i][j];
		}
	}

	FVector Pos04;
	Pos04.X = T.M[0][3];
	Pos04.Y = T.M[1][3];
	Pos04.Z = T.M[2][3];

	FMatrix rot47;
	FRotationMatrix ROT(rot);
	rot47 = rot041.GetTransposed() * ROT;

	if (fabs(rot47.M[0][2]) > 0.000001)
	{
		if (rot47.M[0][2] > 0)
			JointAngle[4] = atan(rot47.M[1][2] / rot47.M[0][2]);
		else
			JointAngle[4] = atan(rot47.M[1][2] / rot47.M[0][2]) + PI;
	}

	if (fabs(rot47.M[2][0]) > 0.000001)
	{
		if (rot47.M[2][0] > 0)
			JointAngle[6] = atan(-rot47.M[2][1] / rot47.M[2][0] + PI);
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
}
~~~
***
# Error

![Joint1 simulation](/images/unreal-engine-inverse-kinematics-1/60788473-d2c88100-a197-11e9-9e7f-90ee0f1f8862.png)
![wristPos simulation](/images/unreal-engine-inverse-kinematics-1/60788472-d2c88100-a197-11e9-9b4c-1f82a41099c8.png)

~~~C++
Vec3f wristPos = pos - rot*Vec3f(0, 0, D[6]); WristPos=wristPos;
FVector wristPos = pos - rot.RotateVector(FVector(0, 0, D[6])); //D[6] = 110
~~~

+ `wristPos`에 `error`가 있다
+ `Debugging`이 필요하다