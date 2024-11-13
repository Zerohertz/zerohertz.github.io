---
title: Machine Learning
date: 2021-09-01 15:14:11
categories:
- 5. Machine Learning
tags:
- Statistics
---
# Theorical Background

+ Types of Machine Learning
  + Supervised Learning
  + Unsupervised Learning
  + Reinforcement Learning
  + Self Learning
  + Feature Learning
  + Sparse Dictionary Learning
  + Anomaly Detection
  + Association Rules
+ Two Supervised Learning Methods
  + Regression
  + Classification
+ Empirical Risk Minimization
  + Loss and Risk Functions
  + Algorithms

<!-- More -->

## Supervised Learning

> A problem of inferring, estimating or learning a function $f$ for explaining $y$ based on $f(x)$

+ $y\in \mathcal{Y}$: Output (Dependent, Response Variable or Label)
+ $x\in \mathcal{X}$: Input (Independent, Predictive Variable of Feature)
+ $f\in \mathcal{F}$: $\mathcal{X}\rightarrow \mathcal{Y}$: Model (Regression Function or Hypothesis, Machine)

> c.f., Unsupervised Learning

## Regression & Classification

+ Regression
  + $\mathcal{Y}$: Infinite and Ordered ($\mathcal{Y}=\mathbb{R},\mathbb{R}^+,\ \mathbb{N}\cup\lbrace0\rbrace,\ ...$)
  + Linear Regression: $\mathcal{Y}=\mathbb{R}$
  + Poisson Regression: $\mathcal{Y}=\mathbb{N}\cup\lbrace0\rbrace$
+ Classification
  + $\mathcal{Y}$: Finite and Unordered ($\mathcal{Y}=\lbrace0,\ 1\rbrace,\ \lbrace-1,\ 1\rbrace,\ \lbrace1,\ ...,\ k\rbrace,\ ...$)
  + Logistic Regression: $\mathcal{Y}=\lbrace0,\ 1\rbrace$
  + Support Vector Machine: $\mathcal{Y}=\lbrace-1,\ 1\rbrace$
  + Multinomial Regression: $\mathcal{Y}=\lbrace1,\ ...,\ k\rbrace$
+ Prediction Based on $x$
  + Regression: $y_x=f(x)$
  + Classification: $y_x=\phi(x)$
    + $\phi(x)$: Classifier as a Function of $f(x)$
    + e.g., $\phi(x)=sign(f(x))$

## Loss, Risk and Empirical Risk Minimization

+ Loss (Cost) Function: $L(y,\ f(x))$
  + It measures `loss` or `cost` for the use of $f$
  + Regression: $y\in \mathbb{R}$
    + Square: $(y-f(x))^2$
    + Quantile: $\rho_\tau(y-f(x)),\ 0<\tau<1$
    + Huber: $H_\delta(y-f(x)),\ \delta>0$
  + Classification: $y\in {-1,\ 1}$
    + Zero-One: $I[y\neq\phi]=I[yf(x)<0]$
    + Logistic: $log\lbrace1+exp(-yf(x))\rbrace$
    + Exponential: $exp(-yf(x))$
    + Hinge: $\max\lbrace0,\ 1-yf(x)\rbrace$
    + $\psi$: $I[yf(x)\leq0]+\max\lbrace0,\ 1-yf(x)\rbrace I[yf(x)>0]$
+ Risk Function: $R(f)=E_{(y,\ x)}L(y,\ f(x))$
  + Long-Term average of losses or costs
    + Square Loss: $E(y-f(x))^2$
    + Zero-One Loss: $EI[y\neq \phi(x)]=P[y\neq \phi(x)]$
    + Condition Risk: $R(f)=E_{y|x}[L(y,\ f(x))|x]$
+ Learning $f\in\mathcal{F}$
  + Learning (Estimating, Training) an optimal
  $f\in\mathcal{F}$ or $R(f)=E_x\lbrace E\_{y|x}L(y,\ f(x))|x\rbrace\geq R(\hat f),\ \forall f\in\mathcal{F}$
  + Example)
    + Square loss with $f(x)=x^T\beta$
      + $\hat\beta=E(xx^T)^{-1}E(xy)$
    + Zero-one loss with $\phi(x)$
      + $\hat\phi(x)=arg\max_kP(y=k|x)$
        + $\hat\phi$: Bayes Classifier
        + $R(\hat\phi)$: Bayes Risk
+ Empirical Risk Minimization:
$\hat f=arg\min_{f\in \mathcal{F}}R_n(f)$
+ Empirical Risk Function:
  $R_n(f)=\Sigma^n_{i=1}L(y_i,\ f(x_i))/n$
  + Training Sample ($y_i,\ x_i$), $i\leq n$:
    Independent samples / observations of $(y,\ x)$
  + Linear Regression with Square Loss:
    $\hat \beta =arg\min_\beta\Sigma^n_{i=1}(y_i-x^T_i\beta)^2/n$
  + Logistic Regression with Logistic Loss:
    $\hat\beta=arg\min_\beta\Sigma^n_{i=1}\lbrace-y_ix^T_i\beta+log(1+exp(x^T_i\beta))\rbrace/n$

***

# Iterative Algorithms

+ Univariate Function Minimization
  + Iterative Grid Search Algorithm
  + Golden Section Algorithm
+ Multivariate Function Minimization
  + Coordinate Descent Algorithm
  + Gradient Descent Algorithm

## Univariate Function Minimization

+ Problem: find $x^*=arg\min_{x\in[a,\ b]}f(x)$
+ Iterative Algorithm: for $n=1,\ 2,\ ...$
  + $x^*\in[a_{n+1},\ b_{n+1}]\subset[a_n,\ b_n]$
+ (Lemma) $f$: convex on $[a_n,\ b_n]$
  + $
\begin{equation}
\left.
\begin{gathered}x^*\in[a_n,\ b_n]\newline \exists_C\in[a_{n+1},\ b_{n+1}]\subset[a_n,\ n_n]\newline s.t.\ \ f(a_{n+1})>f(c)<f(b_{n+1})
\end{gathered}
\right\rbrace\rightarrow x^\ast [a_{n+1},\ b_{n+1}] \notag
\end{equation}
$

### Grid Search

+ Non-Iterative Grid Search Alhorithm: $x^*\approx arg\min_{z\in\lbrace z_k=a+k(b-a)/m|k=0,\ ...,\ m\rbrace}f(z)$
+ Iterative Grid Search Algorithm: for $n=1,\ 2,\ ...$
  + $
\begin{equation}
\left\lbrace
\begin{gathered}
a_{n+1}=a_n+(k_n-1)(b_n-a_n)/m \newline
b_{n+1}=a_n+(k_n+1)(b_n-a_n)/m
\end{gathered}
\right. \notag
\end{equation}
$
  + $k_n=arg\min_{k\in\lbrace0,\ ...,\ m\rbrace}f(a_n+k(b_n-a_n)/m)$

### Golden Section Algorithm

+ Golden Section Algorithm: for $n=1,\ 2,\ ...$
  + Two inner points
    + $
\begin{equation}
\left.
\begin{gathered}
c_n=\gamma a_n+(1-\gamma)b_n \newline
d_n=\gamma b_n+(1-\gamma)a_n
\end{gathered}
\right. \notag
\end{equation}
$
    + Golden Ratio: $\gamma=(\sqrt{5}-1)/2\approx0.618034$
    + $[c_n,\ d_n]\subset[a_n,\ b_n]$
  + Update Rule
    + $
\begin{equation}
\left.
\begin{gathered}
f(c_n)>f(d_n)\rightarrow[a_{n+1},\ b_{n+1}]=[c_n,\ b_n] \newline
f(c_n)<f(d_n)\rightarrow[a_{n+1},\ b_{n+1}]=[a_n,\ d_n]
\end{gathered}
\right. \notag
\end{equation}
$

## Multivariate Function Minimization

+ Problem: $x^*=arg\min_{x\in\mathbb{R}^p}f(x)$
+ Iterative Algorithm: for $n=1,\ 2,\ ...$
  + $x_{n+1}=x_n+\alpha_nd_n,\ \alpha_n>0,\ f(x_{n+1})<f(x_n)$
  + $\alpha_n\in\mathbb{R}$: Step Size or Learning Rate
  + $d_n\in\mathbb{R}^p$: Descent Direction Vector
+ How to find $\alpha_n$ and $d_n$?
  1. Find a descent direction $d_n$
  2. Find a step size $\alpha_n$
     + $a_n=arg\min_{t>0}g_n(t),\ g_n(t)=f(x_n+td_n)$

### Descent Condtion for Direction Vector

+ Direction Opposite to Current Gradient
  + Let $x_\alpha=x+\alpha d$, then by the first order Taylor's expansion around $x$, we have $f(x_\alpha)$
    + $\begin{equation}\begin{aligned}f(x_\alpha)&=f(x)+\nabla f(x)^T(x_\alpha-x)+o(||x_\alpha-x||)\newline &=f(x)+\alpha\nabla f(x)^Td+o(\alpha||d||)\newline &=f(x)+\alpha\nabla f(x)^Td+o(\alpha)\end{aligned}\notag\end{equation}$
    + $\nabla f(x)=\partial f(x)/\partial x_j,\ j\leq p$: Gradient vector of $f$ at $x$
  + (Descent Condition) So we have, $f(x_\alpha)-f(x)<\alpha\nabla f(x)^Td<0$, for all sufficiently small $\alpha>0$, if $\nabla f(x)^Td<0$

### Gradient Descent Algorithm

+ General Form with Negative Gradient Direction
  + $x_{n+1}=x_n+\alpha_nd_n=x_n-\alpha_nD_n\nabla f(x_n)$
    + $D_n$: Strictly Positive Definite Matrix
  + $\nabla f(x_n)^TD_n\nabla f(x_n)>0$
+ Various Directions Vectors
  + Steepest Descent: $D_n=I_p\rightarrow x_{n+1}=x_n-\alpha_n\nabla f(x_n)$
  + Newton-Raphson: $D_n=\nabla^2f(x_n)^{-1},\ \alpha_n=1$
  + Modified Newton's Method: $D_n=\nabla^2f(x_0)$ for some $x_0$
  + Diagonally Scaled Steepest Descent: $D_n=[diag(\nabla^2f(x_n))]^{-1}$

#### Step Size of Gradient Descent Algorithm

+ Limited Minimization Rule
  + $f(x_n+\alpha_nd_n)=\min_{\alpha\in[0,\ s]}f(x_n+\alpha d_n)$, for some $s>0$
+ Successive Step Size Reduction (Armijo Rule, Armijo [1996])
  + Find the first integer $m\geq 0$
  that satisfies the stopping rule $f(x_n+r^msd_n)-f(x_n)\leq\sigma r^ms\nabla f(x_n)^Td_n$,
  for some constant $s>0,\ 0<r<1$ and $0<\sigma<1$
  + Practical Choice: Try $\alpha_n=1,\ 1/2,\ 1/4,\ ...$
+ Deterministic Step Size: Small Constant (0.1, 0.01, 0.001)
+ A Sequence: Square summable but not summable ($\propto1/n$), nonsummable diminishing ($\propto1/\sqrt{n}$), ...

### Coordinate Descent Algorithm

+ Successive Minimization of Univariate Functions
+ Iterative Algorithm: for $n=1,\ 2,\ ...$ repeat inner cycling below
  + Inner Cycling: for $j=1,\ 2,\ ...,\ p$
    + $x_{(n+1)j}arg\min_{t\in\mathbb{R}}f(x_{(n+1)1},\ ...,\ x_{(n+1)(j-1)},\ t,\ x_{n(j+1)},\ ...,\ x_{np}$
    + Direction vector for the $j$th inner cycling step
      + $
        \begin{equation}
        d_{nk}=
        \left\lbrace
        \begin{gathered}-sign(\partial f(\tilde x_{nj})/\partial x_j),\ &k=j\newline 0,\  &k\neq j
        \end{gathered}
        \right. \notag
        \end{equation}
        $
      + $\tilde x_{nj}=(x_{(n+1)1},\ ...,\ x_{(n+1)(j=1)},\ x_{nj},\ x_{n(j+1)},\ ...,\ x_{np})$
+ Inner Cycling Minimization
  + Closed Form vs. Univariate Minimization
  + Invariance to the order of coordinates
  + Individual Corrdinate vs. A Block of Coordinates
  + `one-at-a-time` not `all-at-once`
+ Convergence Cases
  + $f$: Convex and Differentiable $\rightarrow$ Always converges to a local minimizer
    + $f(x,\ y)=x^2+y^2+x+10y$
  + $f$: Convex but Not Differentiable $\rightarrow$ May stuck in a point that is not a local minimizer
    + $f(x,\ y)=(x-y)^2+(x+y)^2+|x-y|+20|x+y|$
  + CD algorithm always converges if $f(x)=g(x)+\Sigma^p_{j=1}h_j(x_j)$, where $g$ is convex differentiable and $h_j$ is convex but may not differentiable for all $j\leq p$ [Tseng, 2001]
    + $f(x,\ y)=x^2+y^2+|x|+10|y|$

### Sub-Gradient (Descent) Algorithm

+ $u(x_0)$: Sub-gradient of $f$ at $x_0$ if $f(x)\geq f(x_0)+u(x_0)^T(x-x_0),\ \forall x\in B(x_0,\ \delta)$
+ Sub-Gradient Descent Algorithm: $x_{n+1}=x_n-\alpha_nu_n$
  + $u_n$: Sub-gradient at $x_n$
  + $f$: Convex but may not differentiable
  + Not a descent method
  + Stopping rule by tracing $f^{best}_n=\min\lbrace f(x_1),\ ...,\ f(x_n)\rbrace$
  + Deterministic step size
    + $\alpha_n$: Small constant, square summable but not summable ($1/n$), nonsummable diminishing ($1/\sqrt{n}$), ...

### Stochastic Gradient (Descent) Algorithm

+ Minimizing Empirical Risk Function: $R_n(\beta)=\Sigma^n_{i=1}L(y_i,\ x_i^T\beta)$
  + Batch: Whole Samples
    + $\beta_{t+1}=\beta_t-\alpha_t\Sigma^n_{i=1}\nabla L(y_i,\ x_i^T\beta_t)$
  + Stochastic: One Random Smaple $(y_i,\ x_i)$
    + $\beta_{t+1}=\beta_t-\alpha_t\nabla L(y_i,\ x_i^T\beta_t)$
  + Mini-Batch: A Number of Random Sample $(y_i,\ x_i),\ i\in S$
    + $\beta_{t+1}=\beta_t-\alpha_t\Sigma_{i\in S}\nabla L(y_i,\ x_i^T\beta_t)$

***

# Learning Methods

+ Linear and Logistic Regression
+ High-Dimensional Linear Regression
  + Ridge
  + LASSO
  + SCAD
+ Forward Greedy Methods
  + Forward Selection
  + Stagewise Regression
  + LARS
+ Support Vector Machine
+ Artificial Neural Network
+ Tree and Ensemble
  + Bagging
  + Random Forest
  + Boosting

## Linear and Logistic Regression

### Linear Regression

+ Regression Function and Prediction
  + $f(x)=x^T\beta,\ y_x=f(x)=x^T\beta$
+ Least Square Estimator, LSE: Empirical Risk Function with Square Loss
  + $R_n(\beta)=\Sigma^n_{i=1}(y_i-x_i^T\beta)^2$
+ Algorithm
  + Direct Calculation
    + $\hat\beta=(\Sigma^n_{i=1}x_ix_i^T)^{-1}\Sigma^n_{i=1}x_iy_i$
  + Gradient Descent Algorithm
    + $\beta_{t+1}=\beta_t+\alpha_t\Sigma^n_{i=1}x_i(y_i-x_i^T\beta_t)$
  + Inner Cycling for Coordinate Descent
    + $\beta\_{(t+1)j}=\Sigma^n\_{i=1}x\_{ij}(y\_i-\Sigma\_{k\neq j}x\_{ik}\tilde{\beta}\_{tk}/\Sigma^n\_{i=1}x^2\_{ij})$

#### Other Empirical Risk Functions

+ Generalized Least Square Estimation
  + $R_n(\beta)=\Sigma_{i,\ j}(y_i-x_i^T\beta)\Omega_{ij}(y_j-x_j^T\beta)$
+ Quantile Regression
  + $R_n(\beta)=\Sigma^n_{i=1}\rho_\tau(y_i-x_i^T\beta)$
  + Linear Programming
  + Sub-Gradient Descent
+ Regularization (Penalization) via $J_\lambda$
  + $R_n^\lambda(\beta)=R_n(\beta)+\Sigma^p_{i=1}J_\lambda(|\beta_j|),\ \lambda>0$
  + Coordinate Descent Algorithm
  + Sub-Gradient Descent
  + Convex Concave Procedure

### Logistic Regression

+ Regression Function and Prediction
  + $f(x)=x^T\beta,\ y_x=\phi(x)=I(f(x)>0)$
+ Empirical Risk Function
  + $R_n(\beta)=\Sigma^n_{i=1}(-y_ix_i^T\beta+log(1+exp(x_i^T\beta)))$
+ Algorithm
  + Gradient Descent Algorithm
    + $\beta_{t+1}=\beta_t-(\Sigma^n_{i=1}x_ip_{ti}(1-p_{ti})x^T_i)^{-1}\Sigma^n_{i=1}(p_{ti}-y_i)x_i$

### Statistical Interpretation

+ Maximum Likelihood Estimation
  + Empirical Risk Function: Negative Log-Likelihood Function
  + Linear Regression
    + Normal Distribution Assumption with Known Variance
    + $y_i|x_i\sim N(f(x_i),\ \sigma^2),\ i\leq n$
  + Logistic Regression
    + Bernoulli Distribution Assumption
    + $y_i|x_i\sim B(1,\ exp(f(x))/(1+exp(f(x)))),\ i\leq n$
  + Another Equivalent Form for $y\in \lbrace-1,\ 1\rbrace$
    + Empirical Risk Function with Logistic Loss
    + $R_n(\beta)=\Sigma^n_{i=1}log(1+exp(-y_ix_i^T\beta))$

## High-Dimensional Linear Regression

### Penalized Estimation

+ Penalized Empirical Risk Functrion
  + $R_n^\lambda(f)=R_n(f)+J^\lambda(f)$
  + $R_n$: Empirical Risk Function
  + $f\in\mathcal{F}$: Regression Fuction
  + $J_\lambda$: Penalty Function
  + $\lambda$: Tuning Parameter (Vector)
+ Objectives and Characteristics
  + Regularization
  + Variable Selection
  + High-Dimensional Data Analysis
  + Structure Estimation

### Variable Selection in Linear Regression

+ Penalized Empirical Risk Function for The Square Loss
  + $R_n^\lambda(\beta)=\Sigma^n_{i=1}(y_i-x_i^T\beta)+\Sigma^p_{j=1}J^\lambda(|\beta_j|)$
+ Algorithm: Penalty Functions
  + Ridge: $J^R_\lambda(|\beta|)=\lambda\beta^2$
  + LASSO: $J^L_\lambda(|\beta|)=\lambda|\beta|$
  + Bridge: $J^B_\lambda(|\beta|)=\lambda|\beta|^\nu,\ \nu>0$
  + SCAD: $\nabla\lambda^S_{\lambda,\ a}(|\beta|)=\lambda I(|\beta|<\lambda)+(\lambda-a|\beta|)+a/(a-1)I(|\beta|\geq\lambda)$
  + MC: $\nabla J^M_{\lambda,\ a}(\beta)=(\lambda-\beta/a)_+$
  + Elastic Net: $J^E_{\lambda,\ \gamma}(\beta)=\lambda\beta+\gamma\beta^2$

### Ridge

+ Example) Simple Linear Regression
  + $\begin{equation}\left\lbrace\begin{gathered}R_n(\beta)=\Sigma^n_{i=1}(y_i-x_i\beta)^2=(\beta-3)^2/2\newline R^\lambda_n(\beta)=(\beta-3)^2/2+\lambda\beta^2\end{gathered}\notag\right.\end{equation}$
  + Least Square Estimator
    + $\tilde{\beta}=arg\min_\beta R_n(\beta)=3$
  + Ridge [Hoerl and Knnard, 1970]
    + $\tilde{\beta}^R=arg\min_\beta R^\lambda_n(\beta)=1/(1+2\lambda)\tilde{\beta}$
+ Results
  + Shrinkage or Biased: $|\hat\beta^R|<|\tilde{\beta}|$ for any $\lambda>0$
  + Better Mean Square Error Under Multicollinearity

### LASSO

+ (Cont.) Simple Linear Regression
  + $R^\lambda_n(\beta)=(\beta-3)^2/2+\lambda|\beta|$
  + Least Square Estimator
    + $\tilde{\beta}=arg\min_\beta R_n(\beta)=3$
  + Least Absolute Shrinkage and Selection Operator (LASSO) [Tibshirani, 1996]
+ Results
  + Shrinkage or Biased: $|\hat\beta^L|<|\tilde{\beta}|$ for any $\lambda>0$
  + Sparsity or Variable Selection: $\hat\beta^L=0$ or $\hat\beta^L\neq0$ for some $\lambda$

### SCAD

+ Smoothly Clipped Absolute Deviation (SCAD) [Fan and Li, 2001]
  + $R_n^\lambda=(\beta-3)^2+J_\lambda(|\beta|)$
+ Solution
  + $\begin{equation}\hat\beta\^2=arg\min_\beta R^\lambda_n(\beta)=\left\lbrace\begin{aligned}0,\ &\ |\tilde{\beta}|<\lambda\newline u(\tilde{\beta},\ \lambda),\ &\ \lambda\leq|\tilde{\beta}|<a\lambda\newline\tilde{\beta},\ &\ |\tilde{\beta}|\geq a\lambda\end{aligned}\right.\notag\end{equation}$
+ Results
  + No Shrinkage or Unbiased: $\hat\beta^S=\tilde{\beta}$ for any $\lambda\leq|\tilde{\beta}|/a$
  + Sparsity: $\hat\beta^S=0$ or $\hat\beta^S\neq0$ for some $\lambda$

### Coordinate Descent Algorithm

+ LSE
  + Problem: Minimize $R_n(\beta)=\Sigma^n_{i=1}(y_i-x_i^T\beta)^2/2$
  + Coordinate Function of $\beta_j$
    + $R_{nj}(\beta_j)\propto\lbrace\Sigma^n_{i=1}x_{ij}^2/2\rbrace\beta_j^2-\lbrace\Sigma^n_{i=1}x_{ij}(y_i-\Sigma_{k\neq j}x_{ik}\beta_k)\rbrace\beta_j$
  + Coordinate Descent Algorithm
    + $\tilde{\beta}\_{(t+1)j}=\lbrace\Sigma^n\_{i=1}x\_{ij}(y\_i-\Sigma\_{k\neq j}x\_{ik}\tilde{\beta}\_{tk})\rbrace/\lbrace\Sigma^n\_{i=1}x\_{ij}^2\rbrace$
+ LASSO
  + Problem: Minimize $R^\lambda_n(\beta)=R_n(\beta)+\lambda\Sigma^p_{j=1}|\beta_j|$
  + Coordinate Function of $\beta_j$
    + $R_{nj}^\lambda(\beta_j)\propto R_{nj}(\beta_j)+\lambda|\beta_j|$
  + Coordinate Descent Algorithm [Friedman et al., 2007]
    + $\beta\_{(t+1)j}=soft(\tilde{\beta}\_{(t+1)j},\ \lambda/\Sigma^n\_{i=1}x\_{ij}^2)$
    + Soft-Thresholding Operator
      + $soft(x,\ \lambda)=arg\min_\beta\lbrace(\beta-x)^2/2+\lambda|\beta|\rbrace=sign(x)(|x|-\lambda)_+$
+ SCAD
  + Problem: Minimize $R_n^\lambda(\beta)=R_n(\beta)+\Sigma^p_{j=1}J_S^\lambda(|\beta_j|)$
  + CCCP + CD Algorithm [Kim et al., 2008], [Lee et al., 2016]
    + $\beta_{t+1}=arg\min_\beta U^{\lambda}_n(\beta|\beta_t)$
    + Conbex-Concave Decomposition of The Penalty
      + $J_S^\lambda(|\beta_j|)=\lambda|\beta_j|+(J_S^\lambda|\beta_j|-\lambda|\beta_j|)$
    + Upper Tight Convex Function of $R_n^\lambda$ at $\beta^c$
      + $U^\lambda_n(\beta|\beta^c)=R_n(\beta)+\Sigma^p_{j=1}\lbrace(\nabla J^\lambda_S(|\beta^c_j|)-\lambda sign(\beta^c_j))\beta_j+\lambda|\beta_j|\rbrace$

### Applications

+ High Dimensional Data Analysis
  + Number of Parameters $\geq$ Number of Samples
  + Allow Various Over-Parameterized Models
+ `Regularization` or `Stabilization`
  + Applied together with almost all learning methods
  + Prevents overfit
  + Structure estimation
+ Structure Construction
  + LASSO: $\lambda\Sigma^p_{j=1}|\beta_j|\leftrightarrow\hat\beta_j=0$ or not
  + Fused LASSO [Tibshirani et al., 2005]
    + $\lambda_1\Sigma^p_{j=1}|\beta_j|+\lambda_2\Sigma^p_{j=2}|\beta_j-\beta_{j-1}|$
      + $\lambda_1$: $\hat\beta_j=0$
      + $\lambda_2$: $\hat\beta_k=\hat\beta_{k+1}=...=\hat\beta_{k'}\neq0$
  + Group LASSO [Yuan and Lin, 2006]
    + $\lambda\Sigma^K\_{k=1}|\sqrt{\Sigma^{p\_k}\_{k=1}\beta^2\_{kj}}$
  + Strong Heredity Interaction LASSO [Choi et al.m 2010]
    + $\lambda\Sigma^p_{j=1}|\beta_j|+\lambda_2\Sigma_{j<k}|\eta_{kj}|,\ \eta_{jk}=\delta_{jk}\beta_j\beta_k$
      + $\lambda_1$: $\beta_j=0$
      + $\lambda_2$: $\beta_j=0$ or $\beta_k=0\rightarrow\eta_{jk}=0$
      + $\lambda_3$: $\beta_j\neq0$ and $\beta_k\neq0\rightarrow\eta_{jk}\neq0$

## Forward Greedy Methods

### Forward Selection in Linear Regression

+ Algorithm: Residual-Based Forward Selection
  + Let $\beta=0$
  + For $t=1,\ 2,\ ...$ repeat the followings
    1. Calculate residuals $r_i=y_i-x_i^T\beta^t,\ \ i\leq n$
    2. Find $k=arg\min_j\Sigma_i(r_i-x_{ij}\beta_j)^2$
    3. Find $\beta^{t+1}=arg\min_{\beta^t,\ \beta_k}\Sigma_i(y_i-\Sigma_Ix_{iI}\beta_I^t-x_{ik}\beta_k)^2$
+ Intuition and Result
  + It iteratively includes the variable $x_k$ that is the most correlated with current residual
  + Sequence of Least Square Estimators: $\beta^0\rightarrow\beta^1\rightarrow\beta^2\rightarrow...$

### Forward Stagewise Regression

+ Originality
  + Stepwise Least Squares [Goldberger, 1961]
  + Residual Analysis [Freund et al., 1961]
+ Algorithm
  + Let $\beta=0$ and fix $\lambda^{FSW}>0$ as a small constant
  + For $t=1,\ 2,\ ...$ repeat the followings
    + Calculate residuals $r_i=y_i-x_i^T\beta^t,\ i\leq n$
    + Find $(k,\ \beta_k)=arg\min_{j,\ \beta_j}\Sigma_i(r_i-x_{ij}\beta_j)^2$
    + Update $\beta^{t+1}=\beta^t+\lambda^{FSW}sign(\beta_k)$
+ Result
  + We get a long sequence of regularized estimators because of the choise $\lambda^{FSW}$

### Least Angle Regression

+ Least Angle Regression (LARS) [Efron et al., 2004]
  + The same as the stragewise regression except that $\beta^{t+1}=\beta^t+\lambda^{LARS}sign(\beta_k)$
    + $\lambda^{LARS}$ makes the new residual equally correlated with all the input variables included
    + $\lambda^{LARS}$ has easy closed from w.r.t. current $\beta^t$
+ Some Connection
  + LARS-LASSO: LARS + Deletion Step = LASSO
  + LARS-Stagewise: Stagewise with $\lambda^{FSW}\rightarrow0=$LARS

## Support Vector Machine

+ Support Vector Machine (SVM) for Binary Classification [Vapnik, 1995]
  + Regression Function and Prediction
    + $f(x)=b+x^Tw,\ \ y_x=\phi(x)=sign(f(x))$
  + Pernalized Problem: Pernalized Empirical Risk Function
    + $R_n^\lambda(\beta)=\Sigma_{i=1}^n\lbrace1-y_if(x_i)\rbrace_++\lambda||w||^2,\ \lambda>0$
    + Hinge Loss + Ridge Penalty
    + $x_+=\max\lbrace0,\ x\rbrace$
  + Stochastic: Sub-Gradient Algorithm [Shalev-Shwartz et al., 2007]
    + $w_{t+1}=w_t+\alpha_t\lbrace-\Sigma^n_{i=1}y_ix_iI(y_if(x_i)<1)+2\lambda w_t\rbrace$

### Maximizing Margin

+ Primal Problem: Minimizing $||w||^2/2+\gamma\Sigma^n_{i=1}\xi_i,\ \gamma>0$ subject to
$y_if(x_i)\geq1-\xi_i,\ \xi_i\geq0,\ i\leq n$
  + $\xi_i$: Slack Variable for Quantifying Degree of Missclassification
    + $x_i$ with $\xi_i=0$: Support Vector
    + $x_i$ with $0<\xi_i\leq1$: Correctly Classified Sample
    + $x_i$ with $1<\xi_i$: Incorrectly Classified Sample
  + $2/||w||$ (Margin): Distance between $f$ and support vector
+ Equivalence between primal and penalized problem
$y_if(x_i)\geq1-\xi_i,\ \xi_i\geq0\leftrightarrow\xi_i=\max\lbrace0,\ 1-y_if(x_i)\rbrace$

### KKT Conditions

+ Lagrange Primal Function with Multipliers:
$||w||^2/2+\gamma\Sigma^n_{i=1}\xi_i-\Sigma^n_{i=1}\alpha_i\lbrace y_if(x_i)-(1-\xi_i)\rbrace-\Sigma^n_{i=1}\mu_i\xi_i$
  + Stationarity: $w=\Sigma^n_{i=1}\alpha_iy_ix_i,\ \Sigma^n_{i=1}\alpha_iy_i=0,\ \gamma=\alpha_i+\mu_i,\ i\leq n$
  + Primal and Dual Feasibility: $y_if(x_i)-(1-\xi_i)\geq0,\ \xi_i\geq0,\ i\leq n$
  + Complementary Slackness: $\alpha_i\lbrace y_if(x_i)-(1-\xi_i)\rbrace=0,\ \mu_i\xi_i=0,\ i\leq n$
+ Dual Problem: Minimizing lagrange dual objective function
$\Sigma^n_{i=1}\Sigma^n_{j=1}y_iy_j\alpha_i\alpha_j\langle x_i,\ x_j\rangle-2\Sigma^n_{i=1}\alpha_i$ subject to $\Sigma^n_{i=1}\alpha_iy_i=0,\ -\leq\alpha_i\leq\gamma,\ i=1,\ ...,\ n$
  + Quadratic Programming (QP)
  + CD Algorithm [Sequential Minimal Optimization, Platt, 1998]
    + Relation between $\alpha_i$ and $\alpha_j,\ i\neq j$
      + $y_i\alpha_i+y_j\alpha_j=-\Sigma_{k\neq i,\ j}y_k\alpha_k=c$
    + Univariate optimization from thje inequality $\alpha_i=(c-y_j\alpha_j)y_i$ under constraints $L\leq\alpha_j\leq H$ for some $0\leq L<H\leq\gamma$
+ Slope: Linear Combination of Samples (Stationarity)
  + $w=\Sigma^n_{i=1}\alpha_iy_ix_i$
+ Implicity of Slope
  + $f(x)=x^Tw+b=\Sigma^n_{i=1}\alpha_iy_ix_i^Tx+b$
  + Inner products are required only
  + Classification depends on non-zero $a_i$s, i.e., support vector $x_i$s (Sparsity)

### Non-Linear Regression Function

+ Linear Solution from Optimization
  + $f(x)=\Sigma^n_{i=1}\alpha_iy_i\langle x_i,\ x\rangle+b$
+ Non-Linear Feature Transformation
  + $x\rightarrow\Phi(x),\ f(x)=\Sigma^n_{i=1}\alpha_iy_i<\Phi(x_i),\ \Phi(x)>+b$
  + Kernel Trick: Kernel Function $K$
    + $K(x,\ x')=\langle\Phi(x_i),\ \Phi(x')\rangle\rightarrow f(x)=\Sigma^n_{i=1}\alpha_iy_iK(x_i,\ x)$
      + Linear Kernel: $K(x,\ x')=\langle x,\ x'\rangle$
      + D-th Order Polynomial Kernel: $K(x,\ x')=(\langle x,\ x'\rangle+I)^d$
      + Gaussian Kernel: $K(x,\ x')=exp(-||x-x'||^2/\sigma^2)$
      + Sigmoid Kernel: $K(x,\ x')=tanh(\kappa_1\langle x,\ x'\rangle+\kappa_2)$
+ SVM as a penalized estimation $R_n^\lambda(\beta)=\Sigma^n\_{i=1}(1-y\_if(x_i))\_++\lambda||f||^2_{\mathcal{H}_K},\ \lambda>0$
  + $\mathcal{H}_K$: Reproducing kernel Hilbert space w.r.t. a kernel $K$
  + Representation Theorem: The minimizer of $R_n$ is represented as $f(x)=\Sigma^n_{i=1}\alpha_iy_iK(x_i,\ x)$
    + Example)
      + $\begin{equation}\left\lbrace\begin{gathered} K(x,\ x')=\langle\Phi(x),\ \Phi(x')\rangle\newline\mathcal{H}\_K=\lbrace\Phi(x)^Tw+b|w\in\mathbb{R}^q,\ b\in\mathbb{R}\rbrace\end{gathered}\right.\notag\rightarrow||f\_{\mathcal(H)\_K}||^2=||w||^2\end{equation}$

## Artificial Neural Network

+ Regression function with one hidden layer
  + $y\in\mathcal{Y}\subset\mathbb{R}^k,\ f(x)=g(a(h(x)))$
  + $h$: Non-linear transformation from $x$ to hidden units via an activation function $\sigma$
    + $h(x)=(h_1(x),\ ...,\ h_m(x))^T,\ h_j(x)=\sigma(x^Tw_j+b_j),\ j\leq m$
    + Sigmoid: $\sigma(t)=1/(1+exp(-t))$
    + ReLu: $\sigma(t)=max\lbrace0,\ t\rbrace$
    + RBF: $\sigma(t)=exp(-v^2/2)$
  + $a$: Linear combinations of hidden units
    + $a(h)=(a_1(h),\ ...,\ a_k(h))^T,\ a_s(h)=h^Tv_s+c_s,\ s\leq k$
  + $g$: Transformation from $a$ to outputs via output functions $g_s,\ s\leq k$
    + $g(a)=(g_1(a),\ ...,\ g_k(a))^T$
    + Regression, Identity: $g_s(a)=a_s$
    + Classification, Softmax: $g_s(a)=exp(a_s)/\Sigma^k_{I=1}exp(a_I)$

### Empirical Risk Function

+ $R_n(\theta)=\Sigma^n_{i=1}L(y_i,\ f(x_i))$
+ $(p+1)\times m+(m+1)\times k$ Parameters
  + $\theta=(W,\ b,\ V,\ c)$
  + $W=(w_1,\ ...,\ w_m)^T,\ b=(b_1,\ ...,\ b_m)^T$
  + $V=(v_1,\ ...,\ v_k)^T,\ c=(c_1,\ ...,\ c_k)^T$
+ Loss Function
  + Regression: $L(y,\ f(x))=\Sigma^k_{s=1}(y_s-f_s(x))^2$
  + Classification: $L(y,\ f(x))=-\Sigma^k_{s=1}\lbrace y_slog(f_s(x))\rbrace$
+ Non-Convex (Non-Differentiable) Optimization
+ (Sub) Gradient Descent Algorithm
  + $\theta_{t+1}=\theta_t-\alpha_t\partial R_n(\theta_t)/\partial\theta$

### Gradient Calculation (Backpropagation)

+ Risk Function Structure
  + $R=L(y,\ f)$
  + $f=g(a)$
  + $a=Vh+b$
  + $h=(\sigma(x^Tw_j+b_j),\ j\leq m)^T$
+ Chain Rule w.r.t. $v_{sl}$
  + $\partial R/\partial v_{sl}=\nabla_fL\times\nabla_ag\times\nabla_{v_{sl}}a=\lbrace\nabla_fL\times\nabla_{a_s}g\rbrace\times h_I$
+ Chain Rule w.r.t. $w_{jt}$
  + $\partial R/\partial w_{jt}=\nabla_fL\times\Sigma^k_{s=1}\nabla_{a_s}g\times v_{sj}\times\nabla_{w_{jt}}h_j$
  $=\lbrace\Sigma^k_{s=1}\lbrace\nabla_fL\times\nabla_{a_s}g\rbrace\times v_{sj}\times\sigma'(x^Tw_j+b_j)\rbrace x_t$

## Tree and Ensemble

### Decision Tree

+ Regression Function
  + $f(x)=\Sigma^M_{m=1}f_m(x)I[x\in R_m]$
  + $M$: Numbers of Rectangles
  + $R_m,\ m\leq M$: Disjoint Rectangular Areas in $\mathbb{R}^p$ (Terminal Nodes)
  + $f_m(x),\ m\leq M$: Local Regression Functions of Interest
+ Empirical Risk Function with A Loss $L$
  + $R_n(f_m,\ R_m,\ m\leq M)=\Sigma^n_{i=1}L(y_u,\ f(x_i))=\Sigma^M_{m=1}\Sigma_{x_i\in R_m}L(y_i,\ f_m(x_i))$
+ Top-Down Induction Algorithm
  + Set an initial Tree $f=(f_m,\ R_m,\ m\leq M)$ and repeat the followings
  + Given a tree $f$, construct $\mathcal{F}_f$ that is the set of all possible trees that can be constructed from $f$
  + Find a tree in $\mathcal{F}_f$ that minimizes the empirical risk
  + Stop if there is no gain in emprical risk or some stoping rules are satisfied
    + The Number of Areas, The Largest Area, Depth, ...

#### Growing Regression Tree by CART

+ Regression Function
  + Local Constant: $f_m(x)=c_m$
  + Local Linear: $f_m(x)=x^T\beta_m$
+ Given a tree $f^t=(f_m^t,\ R_m^t,\ m\leq M^t)$
  + Find the set of all possible trees,
  say $\mathcal{F}^t$, by splitting $R_m^t,\ m\leq M^t$
  once: $R_m^t\rightarrow R_m^t\cap[x_j\leq c]\cup R_m^t\cap [x_j>c]$
  + Find $f^{t+1}\in\mathcal{F}^t$ by minimizing $R_n(f_m,\ R_m,\ m\leq M)=\Sigma^n_{i=1}L(y_i,\ f(x_i))$
+ Local estimation of $f$ with square loss
  + $L(y,\ f(x))=\Sigma^M_{m=1}(y-f_m(x))^2/[x\in R_m]$
  + Local Constant: $\hat c_m=\bar y_m$
  + Local Linear: $\hat{\beta}_m=(X_m^TX_m)^{-1}X_m^Ty_m$
  + $X_m$ and $y_m$: The design matrix and target vector in $R^m$
+ Regression Function: $f(x)=\Sigma^M_{m=1}f_m(x)I[x\in R_m]$
  + Voting: $f_m(x)=arg\max_k P(y=k|x\in R_m)$
+ $f^{t+1}\in\mathcal{F}^t$ by minimizing: $R_n(f_m,\ R_m,\ m\leq M)=\Sigma^n_{i=1}L(y_i,\ f(x_i))$
  + Impurity Measure (Loss): $L(y,\ f(x))=\Sigma^M_{m=1}imp_m(y,\ f_m(x))P(x\in R_m)$
+ Impurity Index: $imp_m(y,\ f_m(x))$
  + Miss-Classification Index: $1-\max_kP(y=k|x\in R_m)$
  + Gini Index: $1-\Sigma_kP(y=k|x\in R_m)^2$
  + Entropy Index: $-\Sigma_kP(y=k|x\in R_m)log_2P(y=k|x\in R_m)$
+ Local Estimation of $f$
  + $\hat P(x_i\in R_m)=\Sigma_iI(x_i\in R_m)/n$
  + $\hat f_m(x)=arg\max_k\Sigma_iI(y_i=k,\ x_i\in R_m)/\Sigma_iI(x_i\in R_m)$

#### ECCP

+ Pruning a tree
  + Too simple regression tree may be easy to interpret but fails to have good prediction accuracy, and too complex regression tree will overfit the samples
  + Hence, pruning process is required
  + One popular measure for pruning is the Empirical Cost-Complexity Pruning (ECCP)
+ ECCP
  + Let $f$ be the tree after the growing and stopping steps
  + Given $\alpha\_0$, we find the minimizer $\hat{f}\_\alpha\in\mathcal{F}$ of the ECCP,
  $C\_{\alpha,\ n}(f\_=\Sigma^M\_{m=1}\Sigma\_{x\_i\in R\_m}(y\_i-f\_m(x\_i))^2+\alpha M$
    + $\mathcal{F}$: Set of all trees obtained by pruning $f$ once
    + To determine $\alpha$ we may use some methods such as test error or cross validation
  + Then the final empirical tree (estimator) $\hat{f}$ is the tree after growing, stopping and pruning
+ Pros
  + Simple interpretation
  + Easy handling of continuous and categorical inputs
  + Robust to input outliers
  + Model-free
+ Cons
  + Low prediction accuracy
  + Bad interpretation of large tree
  + Unstable or high variance
  + Bad estimation of non-squared areas

### Ensemble Methods

> Ensemble method is a supervised learning algorithm (technique) for combining many predictive models $\hat{f}\_k,\ k\leq K$ (weak learners trained individually) into one final predictive model $\hat{f}\_{ens}$ (strong learner) in the following way $\hat{f}\_{env}=\Sigma^K\_{k=1}w\_k\hat{f}\_k$ giving higher prediction accuracy

+ The strong learner is known to over-fit the training data more than weak learners theoretically
+ Some ensemble techniques such as the bagging tends to reduce problems related to over-fitting practically
+ Bagging procedure turns out to be a variance reduction scheme, at least for some base procedures
+ Boosting methods are primarily reducing the (model) bias of the base procedure
+ Random forest is a very different ensemble method than bagging or boosting. From the perspective of prediction, random forests is about as good as boosting, and often better than bagging.

#### Bootstrap Samples

+ Roughly speaking, the bootstrapping is a technique of copying the triaining samples many times so that we can enrich the samples
+ The samples from the bootstrapping are called bootstrap samples
+ There are many bootstrapping schemes but, we only consider the bootstrap samples
  + Let $(y_1,\ x_1),\ ...,\ (y_n,\ x_n)$ be $n$ training samples
  + Randomly draw a sample from the training samples $m$ times with replacement, and then we have $(y_1^*,\ x_1^*),\ ...,\ (y_m^*,\ x_m^*)$, the $m$ bootstrap samples
  + We often consider $m=n$, which is like sampling an entirely new training set
+ Not all of the training samples are included in the bootstrap samples, and are included more than oce
  + When $m=n$, about 36.8% of the training samples are left out, for large $n$

#### Bagging

+ Bagging is the first ensemble method [Breiman, 1996]
  + Bagging is an abbreviation of bootstrap aggregating
  + It stabilizes give predictive model by unifying models constructed with bootstrapped samples, enhancing unstable methods to have higher prediction accuracy
    + Unstable: Slightly perturbing the training set can cause signifiant changes in the estimated model
  + Among Leaarning Methods
    + One of the most stable methods is the 1-nearest neighbor method
    + One of the most unstable methods is regression tree
    + Least square estimation is less stable than Huber's estimation
    + Subset selection method is quite unstable
+ Bagging steps: Bagging constructs a unified predictive model as follows
  + Obtain $B$ bootstrap samples of size $m$, $(y_{b1}^*,\ x_{b1}^*),\ ...,\ (y_{bm}^*,\ y_{bm}^*,\ b\leq B$ from the training samples, $(y_1,\ x_1),\ ...,\ (y_n,\ x_n)$ of size $n$
  + Construct $B$ weak predictive models (individually) $\hat{f}_b,\ b\leq B$ from the $B$ bootstrap samples
  + Unify the models into one predictive model $\hat{f}_{bag}$
    + Averaging for regression: $\hat{f}\_{bag}=\Sigma^B\_{b=1}\hat{f}\_b/B$
    + Voting for classification: $\hat{f}\_{bab}=arg\max_k\Sigma^B\_{b=1}I(\hat{f}\_b=k)/B$
+ Breiman [1996] described heuristically the performance of bagging
  + The model variance of the bagged estimator should be equal or smaller than that of the original estimator
  + There can be a drastic variance reduction if the original estimator is unstable
+ Tibshirani (CMU lecture note) discussed some disadvantages of bagging:
  + Loss of interpretability: The final bagged classifier from trees is not a tree, and so we forfeit the clear interpretative ability of a classification tree
  + Computational complexity: We are essentially multiplying the work of growing a single tree by B (especially if we are using the more involved implementation that prunes and validates on the original training data)
+ An example from Ryan Tibshirani (Wisdom of crowds)
  + Here is a simplified setup with $K=2$ classes to help understand the basic phenomenon
  + Suppose that for a given $x$ we have $B$ independent classifies $\hat{f}_b,\ b\leq B$
  + Assume each classifier has a miss-classification rate of $e=0.4$
  + Assume w.l.o.g. that the true class at $x$ is 1 so that $P(\hat{f}_b(x)=2)=0.4$
  + Now, we form the bagged classifier $\hat{f}\_{bag}=arg\max_k\Sigma^B\_{b=1}I(\hat{f}\_b=k)/B$
  + Let $B\_k=\Sigma^B\_{b=1}I(\hat{f}\_b=k)$ be the number of votes for class $k$
  + Notice that $B_2$ is a binomial random variable: $B_2\sim B(B,\ 0.4)$
  + Hence, the miss-classification rate of the bagged classifier is
  $E_2=P(\hat{f}_{bag}(x)=2)=P(B_2>B_1)=P(B_2\geq B/2)$
  + Easy to see $E_2\rightarrow0$ as $B\rightarrow\infty$, in other words, $\hat{f}_{bag}$ has perfect predictive accuracy as $B\rightarrow\infty$
+ What is the problem in this exaple?
  + Of course, the caveat here is independence
  + The classifiers that we use in practice $\hat{f}_b,\ b\leq B$ are clearly not independent, because they are on very similar data sets (Bootstrap samples from the training set)
+ When will Bagging fail?
  + Bagging a good classifier can improve predictive accuracy
  + Bagging a bad one can seriously degrade accuracy

#### Random Forest

> Random Forest [Breiman, 2001] is a way to create a final prediction model by using many decision trees randomly constructed

+ It is a bagging the enforces more randomness
  + Better prediction than bagging
  + Robust to input outliers
  + Easy computation than orignal tree: No pruning, Parallel computing, ...
+ Bagging from trees has a problem
  + The trees produced from bootstrap samples can be highly correlated
    + $X_i\sim(0,\ \sigma^2),\ Cor(X_i,\ X_j)=\rho,\ i,\ j\leq B\rightarrow Var(\bar{X}=\rho\sigma^2+(1-\rho)\sigma^2/B$
+ Random Forest vs. Bagging Trees
  + Constructing random forest
    + Fit a decision tree to different bootstrap samples
    + When growing the tree, use $q<p$ inputs randomly selected in each step
    + Average the trees
  + constructing bagging from trees
    + Fit a decision tree to different bootstrap samples
    + When growing the tree, use $p$ inputs in each step
    + Average the trees
+ Boosting
  + Boosting is similar to bagging in that we combine many weak predictive models
  + But, boosting is quite different to bagging and sometimes can work much better
    + We can see that boosting uses the whole training samples but adapts weights on the training samples
  + The boosting is developed to minimize training error as fast as possible, and known to give higher prediction accuracy
+ AdaBoost: Adaptive Boost
  + The first boosting algorithm is the AdaBoost (Adaptive Boost) algorithm [Freund and Schapire, 1997]
    + Initialize weights $w_i=1/n,\ i\leq n$
    + For $m\leq M$, repeat followings
      + Construct $\hat{f}_m(x)\in\lbrace-1,\ 1\rbrace$ using weights $w_i,\ i\leq n$
      + Calculate $err\_m=\Sigma^n\_{i=1}w\_iI(y\_i\neq\hat{f}\_m(x\_i))/\Sigma^n\_{i=1}w\_i$
      + Set $c_m=log((1-err_m)/err_m)$
      + Update weights $w_i$ by $w_i=w_iexp(c_mI(y_i\neq\hat{f}_m(x_i)))$
    + Construct the final model as $\hat{f}(x)=sign(\Sigma^M_{m=1}c_m\hat{f}_m(x))$
  + The most interesting part in the AdaBoost algorithm is updating the weights
    + If $err_m\leq 1/2$ then $exp(c_m)\geq1$
    + If $y_i\neq\hat{f}_m(x_i)$ then $w_i$ increases, else does not change
    + Larger weights on misclassified observations
  + Forward Stagewise Regression with Square Loss
    + Let $\beta^1=0$ and a small $\lambda>0$
    + For $t=1,\ 2,\ ...$ repeat the followings
      + $(k,\ \beta_k)=arg\min_{j,\ \beta_j}\Sigma_i(y_i-\Sigma_Ix_{iI}\beta^t_I-x_{ij}\beta_j)^2$
      + $\beta^{t+1}_s=\beta^t_s+\lambda sign(\beta_k)I(s=k)$
  + Forward Stagewise Classification Tree with Exponential Loss
    + Set $M$ trees, $T_1,\ ...,\ T_M$
    + Let $\beta^1=0$ and a small $\lambda>0$
    + For $t=1,\ 2,\ ...$ repeat the followings
      + $(k,\ \beta_k)=arg\min_{j,\ \beta_j}\Sigma_iexp(-y_i\Sigma_IT_I(x_i)\beta^t_j-y_i\beta_jT_j(x_i))$
      + $\beta^{t+1}_s=\beta^t_s+\lambda sign(\beta_k)I(s=k)$

***

# Model Comparison

## Training-Validation Procedure

+ Way of obtaining final model by selecting extra parameter, say $\lambda$, in single learning method based on given samples $\mathcal{S}$
  + Step 1: Preparedifferent $\lambda\_s$,say $\lambda\_k$, $k\leq K$
  + Step 2: Divide $\mathcal{S}$ into $\mathcal{S}\_{tr}\cup\mathcal{S}\_{vd}$ at random.
  + Step 3 (Training): Learn models $f^{\lambda\_k},\ k\leq K$ by using $\mathcal{S}\_{tr}$
  + Step 4 (Validation): Calculate prediction errors by using $\mathcal{S}\_{vd}$
    + $err(f^{\lambda\_k}=\Sigma\_{i\in\mathcal{S}\_{vd}}L(y\_i,\ f^{\lambda\_k}(x\_i)),\ k\leq K$
  + Step 5: Select optimal $\lambda=\lambda^o$
    + $o=arg\min\_kerr(f^{\lambda\_k})$
  + Step 6: Learn final model $f^{\lambda\_k}$ by using the whole $\mathcal{S}$
+ Randomization
  + You may repeat from Step 1 to Step 4 sufficiently many times obtaining averages of validation errors
  + When the sample size is small try cross validation error
+ Complexity Control
  + The training-validation procedure can sometimes be a way of avoiding over-fit although not the best way: small training error but high test error
  + Some learning methods have their own way or empirical experience of selecting extra parameters

## Training-Validation-Test Procedure

+ way of comparing two different learning methods, say $f$ and $g$ based on given samples $\mathcal{S}$
  + Step 1: Divide $\mathcal{S}$ into $\mathcal{S}\_{tr}\cup\mathcal{S}\_{vd}\cup\mathcal{S}\_{ts}$ at random
  + Step 2: Do training-validation procedure based on $\mathcal{S}$ into $\mathcal{S}\_{tr}\cup\mathcal{S}\_{vd}$ for each method to obtain final models, say $f^o$ and $g^o$
  + Step 3 (Test): Calculate prediction errors of $f^o$ and $g^o$ by using $\mathcal{S}\_{st}$
  + Step 4: The winner is who has smaller test error
+ Randomization
  + You may repeat from step 1 to step 4 sufficiently many times, obtaining averages of test errors
