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

![init-1](/images/matlab-app-designer/init-1.png)

![init-2](/images/matlab-app-designer/init-2.png)

<!-- More -->

> 속성 추가

<img src="/images/matlab-app-designer/property.png" alt="property" width="407" />

~~~Matlab 
properties (Access = private)
    Camera = webcam;
end
~~~

> UIAxes 추가

![uiaxes](/images/matlab-app-designer/uiaxes.png)

> Callback 추가

<img src="/images/matlab-app-designer/callback.png" alt="callback" width="294" />

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

<img src="/images/matlab-app-designer/init.png" alt="init" width="752" />
<img src="/images/matlab-app-designer/normal.png" alt="normal" width="752" />
<img src="/images/matlab-app-designer/fault.png" alt="fault" width="752" />

## DAQ App for PHM by ML

> Reference : [LiveDataAcquisition.mlapp](https://uk.mathworks.com/help/daq/live-data-acquisition-app.html)

[Source Code on Github](https://github.com/Zerohertz/DAQ-App-for-PHM)

<img src="/images/matlab-app-designer/ml.png" alt="ml" width="1341" />

## DAQ App for PHM by DNN

> Reference : [LiveDataAcquisition.mlapp](https://uk.mathworks.com/help/daq/live-data-acquisition-app.html)

[Source Code on Github](https://github.com/Zerohertz/DAQ-App-for-PHM)

<img src="/images/matlab-app-designer/dnn.png" alt="dnn" width="1341" />
