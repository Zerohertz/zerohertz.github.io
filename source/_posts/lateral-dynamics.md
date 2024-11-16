---
title: The Lateral Dynamics of a Moving Web
date: 2022-02-17 13:58:40
categories:
- Etc.
tags:
- Mechanical Engineering
---
# Introduction

> 사행이란 무엇인가...

<!-- More -->

# John Jarvis Shelton

> Sign Conventions for Mechanics Analysis.

<img src="/images/lateral-dynamics/sign-conventions-for-mechanics-analysis..png" alt="sign-conventions-for-mechanics-analysis." width="884" />

> Freebodies and Symbols for Steady-State Analysis.

<img src="/images/lateral-dynamics/freebodies-and-symbols-for-steady-state-analysis..png" alt="freebodies-and-symbols-for-steady-state-analysis." width="807" />

+ $Q$: Shear parallel to $y$ axis
+ $N$: Shear normal to web

If deflections are small, the tesnion $T$ of "A" and "B" are approximately equal.

+ "A": Moment equilibrium at Point $O$

<div style="overflow: auto;">

$$M+Qdx-(M+dM)-T\frac{dy}{dx}dx=0$$
</div>
<div style="overflow: auto;">

$$Qdx-dM-Tdy=0$$
</div>
<div style="overflow: auto;">

$$Q=\frac{dM}{dx}+T\frac{dy}{dx}$$
</div>
<div style="overflow: auto;">

$$\frac{dM}{dx}=-EIy'''(\because M=-EIy'')$$
</div>
<div style="overflow: auto;">

$$Q=-EIy'''+T\frac{dy}{dx}$$
</div>
<div style="overflow: auto;">

$$Q=Constant(\because Side\ load\ w = 0)\rightarrow -EIy''''+Ty''=0$$
</div>
<div style="overflow: auto;">

$$\therefore y''''-\frac{T}{EI}y''=y''''-K^2y''=0(K^2=\frac{T}{EI})$$
</div>

+ Beam Theory

$$
\frac{\partial^4y}{\partial x^4}-K^2\frac{\partial^2y}{\partial x^2}=0
$$
$$
N=Q-T\frac{dy}{dx}
$$

+ General Solution

<div style="overflow: auto;">

$$
y(x)=C_1\sinh{Kx}+C_2\cosh{Kx}+C_3x+C_4
$$
</div>

Note that need 4 boundary conditions for this solution.

## Static Model

1. $y(0)=0\rightarrow C_2=-C_4$
2. Upstream roller에서 web과 roller 사이의 마찰이 충분하여 circumferential slip과 lateral slip이 없다면 $y'(0)=0$을 만족한다.
   + Normal Entry Law: 기울기가 0이므로 web이 roller에 수직으로 입사
3. At $x=L$ (No Slip Condition을 만족해야 Normal Entry Law 성립)
   + $y'(L)=C_1K\cosh{KL}+C_2K\sinh{KL}+C_3=\theta_L$

<div style="overflow: auto;">

$$C_1K\cosh{KL}+C_2K\sinh{KL}-C_1K=\theta_L$$
</div>
<div style="overflow: auto;">

$$C_2K\sinh{KL}=\theta_L-C_1K\cosh{KL}+C_1K$$
</div>
<div style="overflow: auto;">

$$C_2=\frac{\theta}{K\sinh{KL}}-C_1\frac{\cosh{KL}-L}{\sinh{KL}}$$
</div>

> Web With Moment at Guide Roller.

<img src="/images/lateral-dynamics/web-with-moment-at-guide-roller..png" alt="web-with-moment-at-guide-roller." width="850" />

4. At steady state, moment of guider roll is zero.

<div style="overflow: auto;">

$$M_L=-EIy''=0\rightarrow y''(L)=0$$
</div>
<div style="overflow: auto;">

$$y''(x)=K^2C_1\sinh{Kx}+K^2C_2\cosh{Kx}$$
</div>
<div style="overflow: auto;">

$$y''(L)=0\rightarrow K^2C_1\sinh{KL}+K^2C_2\cosh{KL}=0$$
</div>
<div style="overflow: auto;">

$$\therefore C_2=-C_1\frac{\sinh{KL}}{\cosh{KL}}$$
</div>

<div style="overflow: auto;">

5. $C_2=\frac{\theta_L}{K\sinh{KL}}-C_1\frac{\cosh{KL}-1}{\sinh{KL}}(\because 3.)$, $C_2=-C_1\frac{\sinh{KL}}{\cosh{KL}}(\because 4.)$
</div>

<div style="overflow: auto;">

$$
\frac{\theta_L}{K\sinh{KL}}-C_1\frac{\cosh{KL}-1}{\sinh{KL}}=-C_1\frac{\sinh{KL}}{\cosh{KL}}
$$
</div>
<div style="overflow: auto;">

$$
\theta_L=C_1\frac{(\cosh{KL})^2-\cosh{KL}-(\sinh{KL})^2}{\cosh{KL}\times\sinh{KL}}\times K\sinh{KL}
$$
</div>
<div style="overflow: auto;">

$$
\therefore C_1=-\frac{\theta_L}{K}\times\frac{\cosh{KL}}{\cosh{KL}-1}
$$
</div>
<div style="overflow: auto;">

$$
C_2=-C_1\frac{\sinh{KL}}{\cosh{KL}}=\frac{\theta_L}{K}\times\frac{\cosh{KL}}{\cosh{KL}-1}\times\frac{\sinh{KL}}{\cosh{KL}}=\frac{\theta_L}{K}\times\frac{\sinh{KL}}{\cosh{KL}-1}
$$
</div>
<div style="overflow: auto;">

$$
C_3=-KC_1=\theta_L\frac{\cosh{KL}}{\cosh{KL}-1}
$$
</div>
<div style="overflow: auto;">

$$
C_4=-C_2=-\frac{\theta_L}{K}\times\frac{\sinh{KL}}{\cosh{KL}-1}
$$
</div>

+ Particular Solution

<div style="overflow: auto;">

$$
y(x)=-\frac{\theta_L}{K}\times\frac{\cosh{KL}}{\cosh{KL}-1}\sinh{Kx}+\frac{\theta_L}{K}\times\frac{\sinh{KL}}{\cosh{KL}-1}\cosh{Kx}
+\theta_L\frac{\cosh{KL}}{\cosh{KL}-1}x-\frac{\theta_L}{K}\times\frac{\sinh{KL}}{\cosh{KL}-1}
$$
</div>

6. Non-dimensionalized by dividing by $L$

<div style="overflow: auto;">

$$
\begin{aligned}
\frac{y}{L}&=\theta_L[-\frac{1}{KL}\frac{\cosh{KL}}{\cosh{KL}-1}\sinh{Kx}\newline
&+\frac{1}{KL}\frac{\sinh{KL}}{\cosh{KL}-1}\cosh{Kx}+\frac{x}{L}\frac{\cosh{KL}}{\cosh{KL}-1}-\frac{1}{KL}\frac{\sinh{KL}}{\cosh{KL}-1}]
\end{aligned}
$$
</div>
<div style="overflow: auto;">

$$
\frac{y}{L}=\theta_L[\frac{\cosh{KL}}{\cosh{KL}-1}(\frac{x}{L}-\frac{\sinh{Kx}}{KL})+\frac{1}{KL}\frac{\sinh{KL}}{\cosh{KL}-1}(\cosh{Kx}-1)]
$$
</div>

7. Let curvature factor $K_c=\frac{y_L}{L\theta_L}$

<div style="overflow: auto;">

$$
\begin{aligned}
K_c&=\frac{\cosh{KL}}{\cosh{KL}-1}(1-\frac{\sinh{KL}}{KL})+\frac{1}{KL}\frac{\sinh{KL}}{\cosh{KL}-1}(\cosh{KL}-1)\newline
\newline
&=\frac{KL\cosh{KL}-(\cosh{KL})(\sinh{KL})+(\sinh{KL})(\cosh{KL})-\sinh{KL}}{KL(\cosh{KL}-1)}
\end{aligned}
$$
</div>
<div style="overflow: auto;">

$$
\therefore Curvature\ factor:\ K_c=\frac{KL\cosh{KL}-\sinh{KL}}{KL(\cosh{KL}-1)}
$$
</div>
$$
\lim_{KL\rightarrow0}K_c=\frac{2}{3}
$$

+ Lateral web deflection from it's original position

<div style="overflow: auto;">

$$
y(x)=\theta_L[\frac{\cosh{KL}}{\cosh{KL}-1}(x-\frac{\sinh{Kx}}{K})+\frac{1}{K}\frac{\sinh{KL}}{\cosh{KL}-1}(\cosh{Kx}-1)]
$$
</div>

+ Angle between web and roller

<div style="overflow: auto;">

$$
\frac{dy}{dx}=\theta(x)=\theta_L[\frac{\cosh{KL}}{\cosh{KL}-1}(1-\cosh{KL})+\frac{\sinh{KL}}{\cosh{KL}-1}\sinh{Kx}]
$$
</div>
<div style="overflow: auto;">

$$
y''=\frac{d^2y}{dx^2}=\theta_L[-\frac{K\cosh{KL}}{\cosh{KL}-1}\sinh{Kx}+\frac{K\sinh{KL}}{\cosh{KL}-1}\cosh{Kx}]
$$
</div>

8. $EIK^2=T$이므로 $EIK=\frac{T}{K}$

+ Bending moment in the web

<div style="overflow: auto;">

$$
\begin{aligned}
M(x)=-EIy''&=\theta_L[EIK\frac{\cosh{KL}}{\cosh{KL}-1}\sinh{Kx}-EIK\frac{\sinh{KL}}{\cosh{KL}-1}\cosh{Kx}]\newline
&=-\frac{T}{K}\theta_L[\frac{\sinh{KL}}{\cosh{KL}-1}\cosh{Kx}-\frac{\cosh{KL}}{\cosh{KL}-1}\sinh{Kx}]
\end{aligned}
$$
</div>
<div style="overflow: auto;">

$$
y'''=\frac{d^3y}{dx^3}=\theta_L[-\frac{K^2\cosh{KL}}{\cosh{KL}-1}\cosh{Kx}+\frac{K^2\sinh{KL}}{\cosh{KL}-1}\sinh{Kx}]
$$
</div>

+ Shear force normal to the elastic curve of the web

<div style="overflow: auto;">

$$
N(x)=-EIy'''=T\theta_L[\frac{\cosh{KL}}{\cosh{KL}-1}\cosh{Kx}-\frac{\sinh{KL}}{\cosh{KL}-1}\sinh{Kx}]
$$
</div>

> Critical Moment Condition

<img src="/images/lateral-dynamics/critical-moment-condition.png" alt="critical-moment-condition" width="441" />

9. Maximum moment occurs at $x=0$ (Upstream roller)

$$
M_0=\frac{-\sinh{KL}}{\cosh{KL}-1}\frac{T}{K}\theta_L
$$

Inside edge에서 $x=0$인 점의 resultant tension을 0으로 만드는 condition을 critical condition이라면,

<div style="overflow: auto;">

$$
(M_0)_{cr}=-\frac{Tw}{6}=-\frac{\sinh{KL}}{\cosh{KL}-1}\frac{T}{K}(\theta_L)\_{cr}
$$
</div>
<div style="overflow: auto;">

$$
(\theta_0)_{cr}=\frac{KL}{6}\frac{w}{L}\frac{\cosh{KL}-1}{\sinh{KL}}
$$
</div>

Downstream roll이 $(\theta_L)_{cr}$만큼의 각도를 가지면 $x=0$에서부터 bending 발생

+ Slack이 발생하지 않는 최대 수정량 (Guide)

<div style="overflow: auto;">

$$
\begin{aligned}
(\frac{y_L}{w})_{cr}&=\frac{KL}{6}K_c\frac{\cosh{KL}-1}{\sinh{KL}}\newline
&=\frac{1}{6}\frac{KL\cosh{KL}-\sinh{KL}}{\sinh{KL}}(\because y_L=K_c\theta_LL)
\end{aligned}
$$
</div>

## Upstream Roller에 $\theta_0$의 각도가 존재하는 Case

<div style="overflow: auto;">
$$
y(0)=0,\ y'(0)=\theta_0,\ y'(L)=\theta_L,\ y''(0)=0
$$
</div>
<div style="overflow: auto;">

$$
\begin{aligned}
\frac{y}{L}&=\theta_L(1-\frac{\theta_0}{\theta_L})[\frac{\cosh{KL}}{\cosh{KL}-1}(\frac{x}{L}-\frac{\sinh{Kx}}{KL})\newline
&+\frac{1}{KL}\frac{\sinh{KL}}{\cosh{KL}-1}(\cosh{Kx}-1)]+(\frac{x}{L})\theta_0
\end{aligned}
$$
</div>
<div style="overflow: auto;">

$$
\frac{y_L}{L\theta_L}=(1-\frac{\theta_0}{\theta_L})[\frac{KL\cosh{KL}-\sinh{KL}}{KL(\cosh{KL}-1)}]+\frac{\theta_0}{\theta_L}
$$
</div>

+ Angle between web and roller

<div style="overflow: auto;">
$$
\begin{aligned}
\frac{dy}{dx}=\theta(x)=&\theta_L(1-\frac{\theta_0}{\theta_L})[\frac{\cosh{KL}}{\cosh{KL}-1}(1-\cosh{Kx})\newline
&+\frac{\sinh{KL}}{\cosh{KL}-1}\sinh{Kx}]+\theta_0
\end{aligned}
$$
</div>

+ Bending moment in the web

<div style="overflow: auto;">

$$
M(x)=\frac{T}{K}(1-\frac{\theta_0}{\theta_L})\theta_L[\frac{\cosh{KL}}{\cosh{KL}-1}(\sinh{Kx})-\frac{\sinh{KL}}{\cosh{KL}-1}\cosh{Kx}]
$$
</div>

+ Shear force normal to the elastic curve of the web

<div style="overflow: auto;">

$$
N(x)=T\theta_L(1-\frac{\theta_0}{\theta_L})[\frac{\cosh{KL}}{\cosh{KL}-1}(\cosh{Kx})-\frac{\sinh{KL}}{\cosh{KL}-1}\sinh{Kx}]
$$
</div>

+ Critical condition인 경우 maximum moment $M_0$ (at $x=0$)

<div style="overflow: auto;">

$$
M_0=-\frac{\sinh{KL}}{\cosh{KL}-1}(\frac{T}{K}\theta_L)(1-\frac{\theta_0}{\theta_L})
$$
</div>

+ Critical guide roller angle

<div style="overflow: auto;">

$$
(\theta_L)_{cr}=\frac{KL}{6}\frac{w}{L}\frac{\cosh{KL}-1}{\sinh{KL}}+\theta_0
$$
</div>

+ Critical correction

<div style="overflow: auto;">

$$
(\frac{y_L}{w})_{cr}=\frac{1}{6}\frac{KL\cosh{KL}-\sinh{KL}}{\sinh{KL}}+\frac{L}{w}\theta_0
$$
</div>