---
title: MATLAB (11)
date: 2019-12-24 16:09:01
categories:
- Etc.
tags:
- MATLAB
- B.S. Course Work
---
# Species classification by sepal

~~~Matlab
clear
 
%Load data
load fisheriris
x=meas(:,1:2);
y=categorical(species);
labels = categories(y);
figure(1)
gscatter(x(:,1),x(:,2),species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
 
%Learning by data
classifier{1}=ClassificationDiscriminant.fit(x,y);
classifier{2}=ClassificationTree.fit(x,y);
classifier{3}=ClassificationKNN.fit(x,y);
%classifier{4}=NaiveBayes.fit(x,y); 2019 ver
classifier_name={'Discriminant Analysis','Classification Tree','Nearest Neighbor'}; %'Naive Bayes'
 
%Check the result
[xx1,xx2]=meshgrid(4:.01:8,2:.01:4.5);
figure(2)
for ii=1:numel(classifier)
    ypred=predict(classifier{ii},[xx1(:) xx2(:)]);
    h(ii)=subplot(2,2,ii);
    gscatter(xx1(:),xx2(:),ypred,'rgb');
    title(classifier_name{ii},'FontSize',15)
    legend off
    axis tight
end
 
%Confusion Matrix
figure(3)
predictResult=predict(classifier{2},x);
y=categorical(y);
predictResult=categorical(predictResult);
plotconfusion(y,predictResult);
~~~
<!-- more -->

![result-1](/images/matlab-11/result-1.png)

![data-point](/images/matlab-11/data-point.png)

![method-of-classification](/images/matlab-11/method-of-classification.png)
***
# ML by example data

~~~Matlab
clear
 
%Load data
load('PracticeData.mat')
 
%Learning by data
classifier{1}=ClassificationDiscriminant.fit(data,Target); %data(:,1:2)로 어떤 데이터 쓸지 결정 가능
classifier{2}=ClassificationTree.fit(data,Target);
classifier{3}=ClassificationKNN.fit(data,Target);
%classifier{4}=NaiveBayes.fit(x,y); 2019 ver
classifier_name={'Discriminant Analysis','Classification Tree','Nearest Neighbor'}; %'Naive Bayes'
 
%Confusion Matrix
figure(1)
predictResult=predict(classifier{1},data);
y=categorical(Target);
predictResult=categorical(predictResult);
plotconfusion(y,predictResult);
 
figure(2)
predictResult=predict(classifier{2},data);
y=categorical(Target);
predictResult=categorical(predictResult);
plotconfusion(y,predictResult);
 
figure(3)
predictResult=predict(classifier{3},data);
y=categorical(Target);
predictResult=categorical(predictResult);
plotconfusion(y,predictResult);
~~~

![result-2](/images/matlab-11/result-2.png)