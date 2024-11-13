---
title: MATLAB (12)
date: 2020-07-17 16:07:02
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# 제어문

## 조건문

### if-elseif-else

~~~Matlab
if 조건1
    실행 문장
elseif 조건2
    실행 문장
elseif 조건3
    실행 문장
...
else
    실행 문장
end
~~~

### switch

~~~Matlab
switch(변수)
    case(값1)
        실행 문장
    case(값2)
        실행 문장
...
    otherwise
        실행 문장
end
~~~

<!-- More -->

## 반복문

### while

~~~Matlab
while(조건)
    실행 문장
end
~~~

### for

~~~Matlab
for 변수 = 시작:끝
    실행 문장
end
~~~

~~~Matlab
for 변수 = 시작:delta:끝
    실행 문장
end
~~~

~~~Matlab
for 변수 = [num1, num2, ...]
    실행 문장
end
~~~

***

# 함수

~~~Matlab
% 주석 작성 시 help 명령어에 출력
function 출력 = 함수명(param1, param2, ...)
실행 문장
...
end
~~~

~~~Matlab
함수명 = @(param1, param2, ...) 공식;
~~~

***

# DNN

~~~Matlab MNIST.m
>> digitDatasetPath=fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset')

digitDatasetPath =

    '/Applications/MATLAB_R2020a.app/toolbox/nnet/nndemos/nndatasets/DigitDataset'

>> imds=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames')

imds = 

  ImageDatastore - 속성 있음:

                       Files: {
                              ' .../toolbox/nnet/nndemos/nndatasets/DigitDataset/0/image10000.png';
                              ' .../toolbox/nnet/nndemos/nndatasets/DigitDataset/0/image9001.png';
                              ' .../toolbox/nnet/nndemos/nndatasets/DigitDataset/0/image9002.png'
                               ... and 9997 more
                              }
                     Folders: {
                              ' .../MATLAB_R2020a.app/toolbox/nnet/nndemos/nndatasets/DigitDataset'
                              }
                      Labels: [0; 0; 0 ... and 9997 more categorical]
    AlternateFileSystemRoots: {}
                    ReadSize: 1
      SupportedOutputFormats: ["png"    "jpg"    "jpeg"    "tif"    "tiff"]
         DefaultOutputFormat: "png"
                     ReadFcn: @readDatastoreImage

>> figure;
perm=randperm(10000,20);
for i=1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
>> layers=[imageInputLayer([28 28 1])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(10)
softmaxLayer
classificationLayer];

>> options=trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',4,'Shuffle','every-epoch','ExecutionEnvironment','cpu','Plots','training-progress') % gpu 가능

options = 

  TrainingOptionsSGDM - 속성 있음:

                    Momentum: 0.9000
            InitialLearnRate: 0.0100
           LearnRateSchedule: 'none'
         LearnRateDropFactor: 0.1000
         LearnRateDropPeriod: 10
            L2Regularization: 1.0000e-04
     GradientThresholdMethod: 'l2norm'
           GradientThreshold: Inf
                   MaxEpochs: 4
               MiniBatchSize: 128
                     Verbose: 1
            VerboseFrequency: 50
              ValidationData: []
         ValidationFrequency: 50
          ValidationPatience: Inf
                     Shuffle: 'every-epoch'
              CheckpointPath: ''
        ExecutionEnvironment: 'cpu'
                  WorkerLoad: []
                   OutputFcn: []
                       Plots: 'training-progress'
              SequenceLength: 'longest'
        SequencePaddingValue: 0
    SequencePaddingDirection: 'right'
        DispatchInBackground: 0
     ResetInputNormalization: 1

>> net=trainNetwork(imds,layers,options)
입력 데이터의 정규화를 초기화하는 중입니다.
｜＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝｜
｜　　Ｅｐｏｃｈ　　｜　　반복　횟수　　｜　　　　경과　시간　　　　　｜　　미니　배치　정확도　　｜　　미니　배치　손실　　｜　　기본　학습률　　｜
｜　　　　　　　　　｜　　　　　　　　　｜　　（ｈｈ：ｍｍ：ｓｓ）　　｜　　　　　　　　　　　　　｜　　　　　　　　　　　　｜　　　　　　　　　　｜
｜＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝｜
｜　　　　　　　１　｜　　　　　　　１　｜　　　　　００：００：００　｜　　　　　　１０．９４％　｜　　　　　２．９６０９　｜　　　０．０１００　｜
｜　　　　　　　１　｜　　　　　　５０　｜　　　　　００：００：０３　｜　　　　　　８１．２５％　｜　　　　　０．５８６６　｜　　　０．０１００　｜
｜　　　　　　　２　｜　　　　　１００　｜　　　　　００：００：０５　｜　　　　　　９４．５３％　｜　　　　　０．１７１９　｜　　　０．０１００　｜
｜　　　　　　　２　｜　　　　　１５０　｜　　　　　００：００：０７　｜　　　　　　９８．４４％　｜　　　　　０．１７８４　｜　　　０．０１００　｜
｜　　　　　　　３　｜　　　　　２００　｜　　　　　００：００：１０　｜　　　　　１００．００％　｜　　　　　０．０６１５　｜　　　０．０１００　｜
｜　　　　　　　４　｜　　　　　２５０　｜　　　　　００：００：１２　｜　　　　　　９８．４４％　｜　　　　　０．０７１３　｜　　　０．０１００　｜
｜　　　　　　　４　｜　　　　　３００　｜　　　　　００：００：１４　｜　　　　　１００．００％　｜　　　　　０．０３２６　｜　　　０．０１００　｜
｜　　　　　　　４　｜　　　　　３１２　｜　　　　　００：００：１５　｜　　　　　１００．００％　｜　　　　　０．０２９１　｜　　　０．０１００　｜
｜＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝｜

net = 

  SeriesNetwork - 속성 있음:

         Layers: [15×1 nnet.cnn.layer.Layer]
     InputNames: {'imageinput'}
    OutputNames: {'classoutput'}

>> Out=classify(net,imds)
~~~

![trainNetwork](https://user-images.githubusercontent.com/42334717/89501255-ff692500-d7fd-11ea-9cf5-9534f547e3c0.png)

~~~Matlab alex.m
camera=webcam;
nnet=alexnet;
while true
    picture=camera.snapshot;
    picture=imresize(picture,[227,227]);
    label=classify(nnet,picture);
    image(picture);
    title(char(label));
    drawnow;
end
~~~