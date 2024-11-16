---
title: Unreal Engine and C++ (1)
date: 2019-01-05 23:18:07
categories:
- Etc.
tags:
- Unreal Engine
- C, C++
---
[개요](https://www.youtube.com/watch?v=V707r4bkJOY)
***
# Class 마법사(Unreal Engine에서 새로운 Class 만들기)

`File` > `New C++ Class` >
![class-지정](/images/unreal-engine-cpp-1/class-지정.png)

<!-- more -->
![이름과-경로-지정](/images/unreal-engine-cpp-1/이름과-경로-지정.png)
![생성-완료](/images/unreal-engine-cpp-1/생성-완료.png)
> MyActor.h

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyActor.generated.h"

UCLASS()
class CPPTEST_API AMyActor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AMyActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
~~~
+ `BeingPalay()`는 액터가 플레이 가능한 상태로 게임에 들어왔음을 알려주는 이벤트
+ `Tick()`는 지난번 들여온 이후의 경과된 시간만큼 프레임당 한 번씩 호출된다

> MyActor.Cpp

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#include "MyActor.h"

// Sets default values
AMyActor::AMyActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AMyActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AMyActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}
~~~
+ `PrimaryActorTick.bCanEverTick = true`를 제거함으로써 필요치 않은 반복 로직을 제거할 수 있다

> 컴파일

![그림의-컴파일-버튼으로-컴파일을-빠르게-할-수-있다](/images/unreal-engine-cpp-1/그림의-컴파일-버튼으로-컴파일을-빠르게-할-수-있다.png)

***
# Class

> 접두사

+ `A` : 스폰가능한 게임플레이 오브젝트의 베이스 클래스에서 확장(Actor, 월드에서 바로 스폰 가능) 
+ `U` : 모든 게임플레이 오브젝트의 베이스 클래스에서 확장(월드에서 바로 인스턴싱 불가, 엑터에 속해야함, 컴포넌트와 같은 오브젝트)

> 선언

~~~C++
UCLASS([specifier, specifier, ...], [meta(key = value, key = value, ...)])
class ClassName : public ParentName // 클래스의 상속
{
    GENERATED_BODY()
}
~~~
+ `UCLASS`매크로에 클래스 지정자나 메타데이터와 같은 지시어가 전달됨
+ `GENERATED_BODY()`매크로는 본문 제일 처음에 와야함
+ 아래의 `Class Specifier`를 위의 `specifier`란에 용도에 맞게 적어서 쓸 수 있다

[Class Specifier](https://docs.unrealengine.com/5.2/ko/gameplay-classes-in-unreal-engine/#%ED%81%B4%EB%9E%98%EC%8A%A4%EC%A7%80%EC%A0%95%EC%9E%90)
***
# 프로퍼티가 에디터에 보이도록 만들기

> MyActor.h

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyActor.generated.h"

UCLASS()
class CPPTEST_API AMyActor : public AActor
{
	GENERATED_BODY()
	
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Damage")
	    int32 TotalDamge;

public:	
	// Sets default values for this actor's properties
	AMyActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
~~~
+ `GENERATED_BODY()`와 `UPROPERTY()`뒤에는 `;`을 붙이지 않는다
+ 아래의 `프로퍼티 지정자`를 `UPROPERTY()`에 적절히 사용함으로써 위의 상황에선 다음과 같은 설정을 할 수 있다
 
![프로퍼티-부여](/images/unreal-engine-cpp-1/프로퍼티-부여.png)
> MyActor.h

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyActor.generated.h"

UCLASS()
class CPPTEST_API AMyActor : public AActor
{
	GENERATED_BODY()
	
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Damage")
		int32 TotalDamage;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Damage")
		float DamageTimeInSeconds;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Transient, Category = "Damage")
		float DamagePerSecond;

public:	
	// Sets default values for this actor's properties
	AMyActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
~~~
![더-많은-프로퍼티-부여](/images/unreal-engine-cpp-1/더-많은-프로퍼티-부여.png)
> MyActor.cpp

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#include "MyActor.h"

// Sets default values
AMyActor::AMyActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	TotalDamage = 200;
	DamageTimeInSeconds = 1.f;

}

// Called when the game starts or when spawned
void AMyActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AMyActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}
~~~
![프로퍼티에-초기값-부여](/images/unreal-engine-cpp-1/프로퍼티에-초기값-부여.png)
***
# 종속적인 프로퍼티 계산 시키기

~~~C++
void AMyActor::PostInitProperties()
{
    Super::PostInitProperties();
    DamagePerSecond = TotalDamage / DamageTimeInSeconds;
}
~~~
# 위와 같이 선언해주면 초기값만 계산한다

> MyActor.h

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyActor.generated.h"

UCLASS()
class CPPTEST_API AMyActor : public AActor
{
	GENERATED_BODY()
	
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Damage")
		int32 TotalDamage;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Damage")
		float DamageTimeInSeconds;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Transient, Category = "Damage")
		float DamagePerSecond;

public:	
	// Sets default values for this actor's properties
	AMyActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	void PostInitProperties();
	void CalculateValues();
	void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent);
};
~~~
> MyActor.cpp

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#include "MyActor.h"

// Sets default values
AMyActor::AMyActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	TotalDamage = 200;
	DamageTimeInSeconds = 1.f;

}

// Called when the game starts or when spawned
void AMyActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AMyActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void AMyActor::PostInitProperties()
{
	Super::PostInitProperties();

	CalculateValues();
}

void AMyActor::CalculateValues()
{
	DamagePerSecond = TotalDamage / DamageTimeInSeconds;
}

#if WITH_EDITOR
void AMyActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	CalculateValues();

	Super::PostEditChangeProperty(PropertyChangedEvent);
}
#endif
~~~
![계산이-되는-모습](/images/unreal-engine-cpp-1/계산이-되는-모습.png)
***
# 여러가지 프로퍼티 지정자

[프로퍼티 지정자](https://docs.unrealengine.com/5.2/ko/unreal-engine-uproperty-specifiers/)