---
title: App Designer
date: 2021-01-20 11:39:36
categories:
- Etc.
tags:
- MATLAB
---
# Plot Cam on App

> 첫 화면

![첫 화면](https://user-images.githubusercontent.com/42334717/105130390-4a67c200-5b2a-11eb-8b6a-82eee5451972.png)

![첫 화면](https://user-images.githubusercontent.com/42334717/105130461-6bc8ae00-5b2a-11eb-9bba-acb327e42bb6.png)

<!-- More -->

> 속성 추가

<img width="407" alt="속성 추가" src="https://user-images.githubusercontent.com/42334717/105130512-8864e600-5b2a-11eb-89ae-b37ed1e2290b.png">

~~~Matlab 
properties (Access = private)
    Camera = webcam;
end
~~~

> UIAxes 추가

![UIAxes](https://user-images.githubusercontent.com/42334717/105130739-edb8d700-5b2a-11eb-849d-dd7b47563d97.png)

> Callback 추가

<img width="294" alt="Callback" src="https://user-images.githubusercontent.com/42334717/105130905-37092680-5b2b-11eb-9345-27949288a1f9.png">

~~~Matlab
function startupFcn(app)
    himg = image(app.UIAxes, zeros(size(snapshot(app.Camera)),'uint8'));
    preview(app.Camera, himg)
end
~~~

***

# GUI for DAQ

## PHM GUI test

> Property로 공구의 상태와 훈련된 모델 지정

~~~Matlab
properties (Access = private)
    state = 'test';
    mod = load('/Users/zerohertz/MATLAB/DAQ-App-for-PHM/TestModel.mat');
end
~~~

> 상태등의 Init

~~~Matlab
function startupFcn(app)
    app.Lamp.Color = 'yellow';
end
~~~

> Slider의 값을 Input으로 지정 및 결과 출력

~~~Matlab
function SliderValueChanged(app, event)
    value = app.Slider.Value;
    app.state = string(app.mod.trainedModel.predictFcn(table(value, 'VariableNames', {'Var1'})));
    if app.state == 'normal'
        app.Lamp.Color = 'blue';
    elseif app.state == 'fault'
        app.Lamp.Color = 'red';
    else
        app.Lamp.Color = 'green';
    end
end
~~~

<img width="752" alt="Init" src="https://user-images.githubusercontent.com/42334717/105429426-c120ce80-5c94-11eb-8abc-30faa58e0eb0.png">
<img width="752" alt="Normal" src="https://user-images.githubusercontent.com/42334717/105429371-a4849680-5c94-11eb-885d-b8c53969ebf7.png">
<img width="752" alt="Fault" src="https://user-images.githubusercontent.com/42334717/105429379-a9494a80-5c94-11eb-8da3-d66d07007368.png">

## DAQ App for PHM by ML

> Reference : [LiveDataAcquisition.mlapp](https://uk.mathworks.com/help/daq/live-data-acquisition-app.html)

[Source Code on Github](https://github.com/Zerohertz/DAQ-App-for-PHM)

<img width="1341" alt="ML" src="https://user-images.githubusercontent.com/42334717/105814056-21e13b80-5ff4-11eb-9ef0-8670de049f43.png">

## DAQ App for PHM by DNN

> Reference : [LiveDataAcquisition.mlapp](https://uk.mathworks.com/help/daq/live-data-acquisition-app.html)

[Source Code on Github](https://github.com/Zerohertz/DAQ-App-for-PHM)

<img width="1341" alt="DNN" src="https://user-images.githubusercontent.com/42334717/105814083-2ad20d00-5ff4-11eb-84f0-4e72a8bd874c.png">
