---
title: Unreal Engine and C++ (2)
date: 2019-02-11 17:20:12
categories:
- Etc.
tags:
- Unreal Engine
- C, C++
---
# 프로젝트 구성

![파일-생성](/images/unreal-engine-cpp-2/파일-생성.png)
![maps-폴더-생성](/images/unreal-engine-cpp-2/maps-폴더-생성.png)

<!-- more -->

![ctrl+s로-map-저장](/images/unreal-engine-cpp-2/ctrl+s로-map-저장.png)
![설정에-들어가서](/images/unreal-engine-cpp-2/설정에-들어가서.png)
![저장한-map을-시작-map으로-저장](/images/unreal-engine-cpp-2/저장한-map을-시작-map으로-저장.png)
![visual-studio의-구성](/images/unreal-engine-cpp-2/visual-studio의-구성.png)
## Hello World

> FPSGameModeBase.h

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "FPSGameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class FPS_API AFPSGameModeBase : public AGameModeBase
{
	GENERATED_BODY()

	virtual void StartPlay() override;
};
~~~

+ `virtual void StartPlay() override;`는 AActor 클래스에서 상속된 `StartPlay()` 함수를 덮어써서 게임플레이가 시작되면 로그 메세지를 출력한다

> FPSGameModeBase.Cpp

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#include "FPSGameModeBase.h"

void AFPSGameModeBase::StartPlay()
{
	Super::StartPlay();

	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Yellow, TEXT("Hello World, this is FPSGameMode!"));
	}
}
~~~

![c++-클래스를-블루프린트로-확장](/images/unreal-engine-cpp-2/c++-클래스를-블루프린트로-확장.png)
![저장](/images/unreal-engine-cpp-2/저장.png)
![시작-모드-설정](/images/unreal-engine-cpp-2/시작-모드-설정.png)
![성공](/images/unreal-engine-cpp-2/성공.png)
> 보다시피 여기서 삽질을 매우 많이했다... `AFPSGameMode::StartPlay()`를 `AFPSGameModeBase::StartPlay()`로 고치는 것을 유념한다...

***
# 캐릭터 임포트

![새-캐릭터-클래스-선언](/images/unreal-engine-cpp-2/새-캐릭터-클래스-선언.png)
![캐릭터-선택](/images/unreal-engine-cpp-2/캐릭터-선택.png)
## 캐릭터 작동 확인

> FPSCharacter.cpp

~~~C++
void AFPSCharacter::BeginPlay()
{
	Super::BeginPlay();
	
	if(GEngine)
	{
		// 5 초간 디버그 메시지를 표시합니다. (첫 인수인) -1 "Key" 값은 이 메시지를 업데이트 또는 새로고칠 필요가 없음을 나타냅니다.
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Red, TEXT("We are using FPSCharacter."));
	}
}
~~~
![build](/images/unreal-engine-cpp-2/build.png)
![캐릭터-클래스-블루프린트로-확장](/images/unreal-engine-cpp-2/캐릭터-클래스-블루프린트로-확장.png)
![캐릭터-기본-설정](/images/unreal-engine-cpp-2/캐릭터-기본-설정.png)
![캐릭터-클래스-작동-확인](/images/unreal-engine-cpp-2/캐릭터-클래스-작동-확인.png)
## W, A, S, D 조작

![wasd](/images/unreal-engine-cpp-2/wasd.png)
![input-설정-1](/images/unreal-engine-cpp-2/input-설정-1.png)

> FPSCharacter.h

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "FPSCharacter.generated.h"

UCLASS()
class FPS_API AFPSCharacter : public ACharacter
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	AFPSCharacter();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	UFUNCTION()
		void MoveForward(float Value);

	UFUNCTION()
		void MoveRight(float value);
};
~~~

> FPSCharacter.cpp

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#include "FPSCharacter.h"

// Sets default values
AFPSCharacter::AFPSCharacter()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AFPSCharacter::BeginPlay()
{
	Super::BeginPlay();
	
	if(GEngine)
	{
		// 5 초간 디버그 메시지를 표시합니다. (첫 인수인) -1 "Key" 값은 이 메시지를 업데이트 또는 새로고칠 필요가 없음을 나타냅니다.
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Red, TEXT("We are using FPSCharacter."));
	}
}

// Called every frame
void AFPSCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void AFPSCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	PlayerInputComponent->BindAxis("MoveForward", this, &AFPSCharacter::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &AFPSCharacter::MoveRight);
}

void AFPSCharacter::MoveForward(float Value) {
	FVector Direction = FRotationMatrix(Controller->GetControlRotation()).GetScaledAxis(EAxis::X);
	AddMovementInput(Direction, Value);
}

void AFPSCharacter::MoveRight(float Value) {
	FVector Direction = FRotationMatrix(Controller->GetControlRotation()).GetScaledAxis(EAxis::Y);
	AddMovementInput(Direction, Value);
}
~~~
> 꿀팁

~~~C++
#include "Engine.h" // 를 추가하면 GEngine관련 명령어에서 오류가 안난다
#include "Components/InputComponent.h" // PlayerInputComponent에서 오류를 없애줌
~~~

## 마우스 카메라

![input-설정-2](/images/unreal-engine-cpp-2/input-설정-2.png)
> `SetupPlayerInputComponent`아래에 선언(FPSCharacter.cpp)

~~~C++
PlayerInputComponent->BindAxis("Turn", this, &AFPSCharacter::AddControllerYawInput);
PlayerInputComponent->BindAxis("LookUp", this, &AFPSCharacter::AddControllerPitchInput);
~~~
> FPSCharacter.cpp

~~~C++
// Fill out your copyright notice in the Description page of Project Settings.

#include "FPSCharacter.h"

// Sets default values
AFPSCharacter::AFPSCharacter()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AFPSCharacter::BeginPlay()
{
	Super::BeginPlay();
	
	if(GEngine)
	{
		// 5 초간 디버그 메시지를 표시합니다. (첫 인수인) -1 "Key" 값은 이 메시지를 업데이트 또는 새로고칠 필요가 없음을 나타냅니다.
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Red, TEXT("We are using FPSCharacter."));
	}
}

// Called every frame
void AFPSCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void AFPSCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	PlayerInputComponent->BindAxis("MoveForward", this, &AFPSCharacter::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &AFPSCharacter::MoveRight);
	PlayerInputComponent->BindAxis("Turn", this, &AFPSCharacter::AddControllerYawInput);
	PlayerInputComponent->BindAxis("LookUp", this, &AFPSCharacter::AddControllerPitchInput);
}

void AFPSCharacter::MoveForward(float Value) {
	FVector Direction = FRotationMatrix(Controller->GetControlRotation()).GetScaledAxis(EAxis::X);
	AddMovementInput(Direction, Value);
}

void AFPSCharacter::MoveRight(float Value) {
	FVector Direction = FRotationMatrix(Controller->GetControlRotation()).GetScaledAxis(EAxis::Y);
	AddMovementInput(Direction, Value);
}
~~~
## Jump 설정

![jump는-action-mapping에서-설정한다](/images/unreal-engine-cpp-2/jump는-action-mapping에서-설정한다.png)

> 함수 선언(FPSCharacter.h)

~~~C++
UFUNCTION()
   	void StartJump();

UFUNCTION()
	void StopJump();
~~~

> 함수 정의(FPSCharacter.cpp)

~~~C++
void AFPSCharacter::StartJump() {
	bPressedJump = true;
}

void AFPSCharacter::StopJump() {
	bPressedJump = false;
}
~~~

> Jump 액션 바인딩(FPSCharacter.cpp의 SetupPlayerInputComponent)

~~~C++
PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &AFPSCharacter::StartJump);
PlayerInputComponent->BindAction("Jump", IE_Released, this, &AFPSCharacter::StopJump);
~~~
## Mesh 추가

![스켈레탈-메시-설정](/images/unreal-engine-cpp-2/스켈레탈-메시-설정.png)
![그림자가-진다](/images/unreal-engine-cpp-2/그림자가-진다.png)
***
# 발사체 구현

![fire-input-설정](/images/unreal-engine-cpp-2/fire-input-설정.png)
## 발사체 클래스 정의

![actor-class-설정](/images/unreal-engine-cpp-2/actor-class-설정.png)
## 발사체의 여러 Component 추가

+ USphere Component

> FPSProjectile.h(FPSProjectile 인터페이스에 USphereComponent로 레퍼런스 추가)

~~~C++
UPROPERTY(VisibleDefaultsOnly, Category = Projectile)
    USphereComponent* CollisionComponent
~~~
> FPSProjectile.cpp의 AFPSProjectile 생성자

~~~C++
// 구체를 단순 콜리전 표현으로 사용
CollisionComponent = CreateDefaultSubobject<USphereComponent>(TEXT("SphereComponent"));
// 구체의 콜리전 반경을 설정
CollisionComponent->InitSphereRadius(15.0f);
// 루트 컴포넌트를 콜리전 컴포넌트로 설정
RootComponent = CollisionComponent;
~~~
+ Movement Component

> FPSProjectile.h

~~~C++
UPROPERTY(VisibleAnywhere, Category = Movement)
    UProjectileMovementComponent* ProjectileMovementComponent;
~~~
> FPSProjectile.cpp의 AFPSProjectile 생성자

~~~C++
ProjectileMovementComponent = CreateDefaultSubobject<UProjectileMovementComponent>(TEXT("ProjectileMovementComponent"));
ProjectileMovementComponent->SetUpdatedComponent(CollisionComponent);
ProjectileMovementComponent->InitialSpeed = 3000.0f;
ProjectileMovementComponent->MaxSpeed = 3000.0f;
ProjectileMovementComponent->bRotationFollowsVelocity = true;
ProjectileMovementComponent->bShouldBounce = true;
ProjectileMovementComponent->Bounciness = 0.3f;
~~~
+ Defining Initial Speed

> FPSProjectile.h

~~~C++
void FireInDirection(const FVector& ShootDirection); // 빌사체의 속도를 발사 방향으로 초기화하는 함수
~~~

> FPSProjectile.cpp

~~~C++
void AFPSProjectile::FireInDirection(const FVector& ShootDirection)
{
    ProjectileMovementComponent->Velocity = ShootDirection * ProjectileMovementComponent->InitialSpeed;
}
~~~
+ 발사 입력 액션 Binding

> FPSCharacter.h

~~~C++
UFUNCTION()
    void Fire(); // 함수 선언
~~~

> FPSCharacter.cpp의 SetupPlayerInputComponent

~~~C++
PlayerInputComponent->BindAction("Fire", IE_Pressed, this, &AFPSCharacter::Fire); // Binding
~~~

> FPSCharacter.cpp

~~~C++
void AFPSCharacter::Fire() // 함수 정의부
{
}
~~~
+ 발사체의 스폰 위치 정의

> FPSCharacter.h

~~~C++
UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Gameplay // BlueprintReadWrite를 통하여 블루프린트 내에서 총구 오프셋의 값을 구하고 설정할 수 있다
    FVector Muzzleoffset; // 카메라 스페이스 오프셋 벡터를 사용해 스폰 위치 결정

UPROPERTY(EditDefaultsOnly, Category = Projectile) // EditDefaultsOnly는 클래스를 블루프린트의 디폴트로만 설정할 수 있다는 뜻
    TSubclassOf<class AFPSProjectile> ProjectileClass;
~~~

## 발사체의 발사 구현

+ Fire 함수 구현

> FPSCharacter.cpp

~~~C++
#include "FPSProjectile.h" // 헤더파일 추가

void AFPSCharacter::Fire() // 함수 정의
{
    // 프로젝타일 발사를 시도합니다.
    if (ProjectileClass)
    {
        // 카메라 트랜스폼을 구합니다.
        FVector CameraLocation;
        FRotator CameraRotation;
        GetActorEyesViewPoint(CameraLocation, CameraRotation);

        // MuzzleOffset 을 카메라 스페이스에서 월드 스페이스로 변환합니다.
        FVector MuzzleLocation = CameraLocation + FTransform(CameraRotation).TransformVector(MuzzleOffset);
        FRotator MuzzleRotation = CameraRotation;
        // 조준을 약간 윗쪽으로 올려줍니다.
        MuzzleRotation.Pitch += 10.0f;
        UWorld* World = GetWorld();
        if (World)
        {
            FActorSpawnParameters SpawnParams;
            SpawnParams.Owner = this;
            SpawnParams.Instigator = Instigator;
            // 총구 위치에 발사체를 스폰시킵니다.
            AFPSProjectile* Projectile = World->SpawnActor<AFPSProjectile>(ProjectileClass, MuzzleLocation, MuzzleRotation, SpawnParams);
            if (Projectile)
            {
                // 발사 방향을 알아냅니다.
                FVector LaunchDirection = MuzzleRotation.Vector();
                Projectile->FireInDirection(LaunchDirection);
            }
        }
    }
}
~~~
![블루프린트로-정의](/images/unreal-engine-cpp-2/블루프린트로-정의.png)
![component-추가](/images/unreal-engine-cpp-2/component-추가.png)
![character-설정](/images/unreal-engine-cpp-2/character-설정.png)
![실행-결과](/images/unreal-engine-cpp-2/실행-결과.png)
## 발사체 콜리전 및 수명 구성

![새-오브젝트-채널-추가](/images/unreal-engine-cpp-2/새-오브젝트-채널-추가.png)
![새-프리셋-추가](/images/unreal-engine-cpp-2/새-프리셋-추가.png)
![새-콜리전-채널-세팅-사용-fpsprojcetile.cpp-](/images/unreal-engine-cpp-2/새-콜리전-채널-세팅-사용-fpsprojcetile.cpp-.png)
+ 수명 추가

> FPSProjectile.cpp

~~~C++
InitialLifeSpan = 3.0f; // 3초 후 죽는다
~~~