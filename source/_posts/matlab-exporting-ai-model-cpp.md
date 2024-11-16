---
title: Exporting AI Model to C++ from MATLAB
date: 2021-02-26 13:15:06
categories:
- 5. Machine Learning
tags:
- MATLAB
- Statistics
- C, C++
---
# MATLAB Coder

> MATLAB Coder를 통해 C 또는 C++ 코드를 생성할 수 있다.

+ Export types
  + 소스 코드
  + 정적 라이브러리
  + 동적 라이브러리
+ Prerequisites for Deep Learning
  + Intel CPUs
    + [Intel Math Kernel Library for Deep Neural Networks (Intel MKL-DNN)](https://www.intel.com/content/www/us/en/developer/topic-technology/open/overview.html)
  + ARM CPUs
    + ARM Compute Library

![MatLAB Coder](https://user-images.githubusercontent.com/42334717/109254668-d78eec80-7835-11eb-897b-228d0a7b468f.png)

<!-- More -->

***

# Exporting Machine Learning Model

~~~Matlab classifyX.m
function label = classifyX(X)
CompactMdl = loadLearnerForCoder('Model');
label = predict(CompactMdl, X);
end
~~~

~~~Matlab test_classifyX.m
num = table(1);

label = classifyX(num);
~~~

~~~C classifyX.c
#include "classifyX.h"
#include "Gaussian.h"
#include "classifyX_types.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"

void classifyX(double X, cell_wrap_0 label[1])
{
  static const char cv[12] = { 'f', 'n', 'a', 'o', 'u', 'r', 'l', 'm', 't', 'a',
    ' ', 'l' };

  double dv[2];
  double svT[2];
  double d;
  int i;
  int k;
  int loop_ub;
  boolean_T b[2];
  boolean_T b_tmp;
  boolean_T exitg1;
  boolean_T y;
  svT[0] = 0.17677669529663687;
  svT[1] = -0.17677669529663687;
  Gaussian(svT, 0.062499999999999986, (X - 0.5) / 0.70710678118654757 / 4.0, dv);
  d = -dv[0] + dv[1];
  b[0] = rtIsNaN(-d);
  b_tmp = rtIsNaN(d);
  b[1] = b_tmp;
  y = true;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 2)) {
    if (!b[k]) {
      y = false;
      exitg1 = true;
    } else {
      k++;
    }
  }

  label[0].f1.size[0] = 1;
  label[0].f1.size[1] = 5;
  for (i = 0; i < 5; i++) {
    label[0].f1.data[i] = cv[i << 1];
  }

  if (!y) {
    if ((-d < d) || (rtIsNaN(-d) && (!b_tmp))) {
      k = 1;
    } else {
      k = 0;
    }

    loop_ub = k + 4;
    label[0].f1.size[0] = 1;
    label[0].f1.size[1] = k + 5;
    for (i = 0; i <= loop_ub; i++) {
      label[0].f1.data[i] = cv[k + (i << 1)];
    }
  }
}
~~~

~~~C main.c
#include "main.h"
#include "classifyX.h"
#include "classifyX_terminate.h"
#include "classifyX_types.h"
#include "rt_nonfinite.h"

static double argInit_real_T(void);
static void main_classifyX(void);

static double argInit_real_T(void)
{
  return 0.0;
}

static void main_classifyX(void)
{
  cell_wrap_0 label[1];
  classifyX(argInit_real_T(), label);
}

int main(int argc, const char * const argv[])
{
  (void)argc;
  (void)argv;

  main_classifyX();

  classifyX_terminate();
  return 0;
}
~~~

***

# Exporting Deep Neural Network Model (CWT-CNN)

~~~Matlab classifyX.m
function label = classifyX(X)
persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork('net.mat');
end
label = classify(net, X);
end
~~~

~~~Matlab test.m
load('test');
testdat = cwt(test);
IMG = uint8(mat2gray(abs(testdat))*255);
GryMat = imresize(IMG, [28 28]);
label = classifyX(GryMat);
~~~

~~~Matlab coder_src.m
cfg = coder.config('lib');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('mkldnn');
cfg.GenCodeOnly(false);

codegen -args {testdat} -config cfg classifyX
~~~

~~~C++ classifyX.cpp
#include "classifyX.h"
#include "DeepLearningNetwork.h"
#include "categorical.h"
#include "classifyX_data.h"
#include "classifyX_initialize.h"
#include "classifyX_internal_types.h"
#include "postProcessOutputToReturnCategorical.h"
#include "predict.h"

static net0_0 net;
static boolean_T net_not_empty;

void classifyX(const unsigned char X[784], coder::categorical *label)
{
  coder::categorical labelsCell[1];
  cell_wrap_7 rv[1];
  cell_wrap_7 r;
  if (!isInitialized_classifyX) {
    classifyX_initialize();
  }

  if (!net_not_empty) {
    coder::DeepLearningNetwork_setup(&net);
    net_not_empty = true;
  }

  coder::DeepLearningNetwork_predict(&net, X, r.f1);
  rv[0] = r;
  coder::DeepLearningNetwork_postProcessOutputToReturnCategorical(rv, labelsCell);
  *label = labelsCell[0];
}

void classifyX_init()
{
  net_not_empty = false;
}
~~~

> Print Result of Test

![Print Result of Test](https://user-images.githubusercontent.com/42334717/109256818-29397600-783a-11eb-9b9a-6727dce26b38.png)

~~~C++ main.cpp
#include "main.h"
#include "categorical.h"
#include "classifyX.h"
#include "classifyX_terminate.h"
#include <iostream>

static void argInit_28x28_uint8_T(unsigned char result[784]);
static unsigned char argInit_uint8_T();
static void main_classifyX();

static void argInit_28x28_uint8_T(unsigned char result[784])
{
  for (int idx0 = 0; idx0 < 28; idx0++) {
    for (int idx1 = 0; idx1 < 28; idx1++) {
      result[idx0 + 28 * idx1] = argInit_uint8_T();
    }
  }
}

static unsigned char argInit_uint8_T()
{
  return 0U;
}

static void main_classifyX()
{
  coder::categorical label;
  unsigned char test1[784] = {test1};
  classifyX(test1, &label);
  unsigned char test2[784] = {test2};
  classifyX(test2, &label);
}

int main(int, const char * const [])
{
  main_classifyX();
  classifyX_terminate();;
  system("PAUSE");
  return 0;
}
~~~