---
title: Development of a Obstructive Sleep Apnea Diagnosis Algorithm using HRV
date: 2021-01-14 11:28:10
categories:
- Etc.
tags:
- MATLAB
- Statistics
---
[Source in Github](https://github.com/Zerohertz/Development-of-a-Obstructive-Sleep-Apnea-Diagnosis-Algorithm-Using-HRV)

# HRV(Heart Rate Variability)

## Detecting R-R Interval

~~~Matlab rrInterval.mat
function [qrspeaks, locs, y] = rrInterval(time, ecg)
t = time;
wt = modwt(ecg, 5);
wtrec = zeros(size(wt));
wtrec(4:5, :) = wt(4:5, :);
y = imodwt(wtrec, 'sym4');
y = abs(y).^2;
[qrspeaks, locs] = findpeaks(y, t, 'MinPeakHeight', 0.1, 'MinPeakDistance', 0.450); %time과 y에 대한 그래프를 해석 후 파라미터 결정
end
~~~

> ECG

<img width="672" alt="ECG" src="https://user-images.githubusercontent.com/42334717/103261452-fbc07f80-49e4-11eb-8181-4251df1a5af5.png">

~~~Matlab
>> load mit200
>> plot(tm, ecgsig)
~~~

<!-- More -->

> R Peaks Localized by Wavelet Transform with Automatic Annotations

<img width="1051" alt="R Peaks Localized by Wavelet Transform with Automatic Annotations" src="https://user-images.githubusercontent.com/42334717/103263042-11847380-49ea-11eb-8daa-b840f7f837a5.png">

~~~Matlab
>> [qr, lo, y] = rrInterval(tm, ecgsig);
>> plot(tm, y)
>> hold on
>> plot(lo, qr, 'ro')
>> plot(tm(ann), y(ann), 'k*')
>> grid on
~~~

## Make HRV Data from R-R Interval

~~~Matlab makeHRV.m
function [time, val] = makeHRV(locat)
for num = (1:length(locat)-1)
    val(num) = (locat(num + 1) - locat(num)) * 1000;
    time(num) = locat(num);
end
end
~~~

> HRV

![HRV](https://user-images.githubusercontent.com/42334717/103266138-4f859580-49f2-11eb-8ca9-ee74b18e45d7.jpg)

~~~Matlab checkRR.m
function [] = checkRR(paramName, size, param)
cd apnea-ecg-database-1.0.0/
wfdb2mat(paramName);
[tm, data] = rdmat(append(paramName, 'm'));
cd ..

ecg = data(:, 1);
tm = tm(1:(length(tm)/10));
ecg = ecg(1:(length(ecg)/10));
ra = fix(rand() * length(tm) / 200);

[qr, lo, y] = rrIntervalparam(tm, ecg, param);
[tt, hh] = makeHRV(lo);

i = 1;

for N = lo
    num = find(tm == N);
    rPeak(i) = ecg(num);
    i = i + 1;
end

fig = figure;
set(fig, 'Position', [0 0 1920 1080])

subplot(3,1,1)
plot(tm, ecg, 'LineWidth', 1)
hold on
plot(lo, rPeak, 'ro')
axis([ra ra + size -1 3])
xlabel('Time(sec)')
ylabel('ECG(mV)')
grid on
set(gca, 'fontsize', 15)

subplot(3,1,2)
plot(tm, y, 'LineWidth', 1)
hold on
plot(lo, qr, 'ro')
axis([ra ra + size 0 0.9])
xlabel('Time(sec)')
ylabel('Amplitude')
grid on
set(gca, 'fontsize', 15)

subplot(3,1,3)
bar(tt, hh, 0.01)
hold on
plot(tt, hh, 'ro')
axis([ra ra + size 0 2000])
xlabel('Time(sec)')
ylabel('HRV(msec)')
grid on
set(gca, 'fontsize', 15)

cd RR/
saveas(gca, append(append(paramName, 'checkRR', string(param)), '.bmp'))
cd ..
end
~~~

***

# Condition Indicators of HRV

> Data 사용

~~~Matlab
wfdb2mat('FileName');
[tm, ecg, Fs, labels] = rdmat('FileName' + 'm');
[apn_tm, apn]= rdann('FileName', 'apn');
~~~

## Time Domain Analysis



## Frequency Domain Analysis

~~~Matlab freqHRV.m
function [f, P1] = freqHRV(hrv, Length, SamplingTime)
y=fft(hrv);
SamplingRate = Length / SamplingTime;
P2 = abs(y/Length);
P1 = P2(1:fix(Length/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f = SamplingRate * (0:(fix(Length/2)))/Length;
end
~~~

~~~Matlab freqHRV1.m
function [f, P1] = freqHRV1(hrv, Length, SamplingTime)
SamplingRate = Length / SamplingTime;
NFFT = 2^(ceil(log2(length(hrv))));
Y = fft(hrv, NFFT) / Length;
f = SamplingRate / 2 * linspace(0, 1, NFFT / 2 + 1);
P1 = 2*abs(Y(1:fix(NFFT/2+1)));
end
~~~

~~~Matlab freqHRVanalysis.m
function [VLF, LF, HF] = freqHRVanalysis(freq, amp)
i = 1;
j = 1;
k = 1;
VLF_amp(1) = 1;
LF_amp(1) = 1;
HF_amp(1) = 1;
for num = (1:length(freq))
    if 0.003 <= freq(num) && freq(num) < 0.04
        %VLF_freq(i) = freq(num);
        VLF_amp(i) = amp(num)^2;
        i = i + 1;
    elseif 0.04 <= freq(num) && freq(num) < 0.15
        %LF_freq(j) = freq(num);
        LF_amp(j) = amp(num)^2;
        j = j + 1;
    elseif 0.15 <= freq(num) && freq(num) <= 0.4
        %HF_freq(k) = freq(num);
        HF_amp(k) = amp(num)^2;
        k = k + 1;
    end      
end
VLF = mean(VLF_amp);
LF = mean(LF_amp);
HF = mean(HF_amp); 
end
~~~

~~~Matlab windowHRV.m
function [f, P] = windowHRV(time, ECG, SamplingRate, Winsize)
%Sampling Rate(Hz)
%Window size(Min)
SamplingTime = Winsize * 60;
Win = fix(SamplingRate * SamplingTime);
num = fix(length(ECG) / Win) - 10;

[~,ll,~]=rrInterval(time, ECG);
[~,bb]=makeHRV(ll);
freqHRVplot(bb, length(bb), length(bb) / 100);

for N = (1:num)
    Time_Arr(:, N) = time(Win * (N - 1) + 1:Win * N);
    ECG_Arr(:, N) = ECG(Win * (N - 1) + 1:Win * N);
end

f = NaN(1000, num);
P = NaN(1000, num);

for N = (1:num)
    [~, lo, ~] = rrInterval(Time_Arr(:, N), ECG_Arr(:, N));
    [winTime, HRV] = makeHRV(lo);
    [fr, P1] = freqHRV(HRV, length(HRV), SamplingTime);
    f(1:length(fr), N) = fr;
    P(1:length(P1), N) = P1;
end
end
~~~

~~~Matlab windowSpO2.m
function [SpO2] = windowSpO2(data, SamplingRate, Winsize)
SamplingTime = Winsize * 60;
Win = fix(SamplingRate * SamplingTime);
num = fix(length(data) / Win) - 10;

for N = (1:num)
    SpO2_Arr(:, N) = data(Win * (N - 1) + 1:Win * N);
end

for N = (1:num)
    SpO2(N) = mean(SpO2_Arr(:, N));
end
end
~~~

~~~Matlab freqHRVCI.m
function [F] = freqHRVCI(f, P)
for N = (1:length(f(1,:)))
[f1, f2, f3] = freqHRVanalysis(f(:, N), P(:, N));
F(N, 1) = f1;
F(N, 2) = f2;
F(N, 3) = f3;
end
end
~~~

~~~Matlab makeTableofHRV.m
function [tab] = makeTableofHRV(tm, data, SamplingRate, Windowsize, Apn)
[f, p] = windowHRV(tm, data(:, 1), SamplingRate, Windowsize);
F = freqHRVCI(f, p);
sp = windowSpO2(data(:, 5), SamplingRate, Windowsize);

tab = table(F(:, 1), F(:, 2), F(:, 3), sp');
tab.Properties.VariableNames{'Var1'} = 'VLF';
tab.Properties.VariableNames{'Var2'} = 'LF';
tab.Properties.VariableNames{'Var3'} = 'HF';
tab.Properties.VariableNames{'Var4'} = 'SpO2';
end
~~~

~~~Matlab makeLabelofHRV.m
function [tab] = makeLabelofHRV(tm, ecg, SamplingRate, Windowsize, tmApn, Apn)
init = tmApn(1);
tm = tm(init:length(tm));
ecg = ecg(init:length(ecg));

[f, p] = windowHRV(tm, ecg, SamplingRate, Windowsize);
F = freqHRVCI(f, p);

for N = (1:length(F)) % 78(Normal) / 65(Apnea)
    apn(N) = mean(Apn(Windowsize * (N - 1) + 1:Windowsize * N));
end

apn = abs(apn - 78) / (78 - 65);
tab = table(F(:, 1), F(:, 2), F(:, 3), apn');
tab.Properties.VariableNames{'Var1'} = 'VLF';
tab.Properties.VariableNames{'Var2'} = 'LF';
tab.Properties.VariableNames{'Var3'} = 'HF';
tab.Properties.VariableNames{'Var4'} = 'apn';
end
~~~

~~~Matlab apneaFreqPlot.m
function [tab] = apneaFreqPlot(paramName, Winsize)
cd apnea-ecg-database-1.0.0/
wfdb2mat(paramName);
[tm, data] = rdmat(append(paramName, 'm'));
[tmApn, apn] = rdann(paramName, 'apn');
cd ..

tab = makeLabelofHRV(tm, data, 100, Winsize, tmApn, apn);

tim = (1:height(tab))*Winsize;


fig = figure;
set(fig, 'Position', [0 0 1920 1080])

subplot(3,1,1)
plot(tim, tab.HF, 'Color', 'black', 'LineWidth', 1.2)
hold on
plot(tim(tab.apn == 0), tab.HF(tab.apn == 0), 'Color', 'blue', 'Marker', 'o', 'LineWidth', 2, 'LineStyle', 'none')
plot(tim(tab.apn > 0), tab.HF(tab.apn > 0), 'Color', 'red', 'Marker', '*', 'LineWidth', 2, 'LineStyle', 'none')
legend('HF', 'Normal', 'Apnea')
xlabel('Time(min)')
ylabel('Amplitude(msec)')
grid on
set(gca, 'fontsize', 15)

subplot(3,1,2)
plot(tim, tab.LF, 'Color', 'black', 'LineWidth', 1.2)
hold on
plot(tim(tab.apn == 0), tab.LF(tab.apn == 0), 'Color', 'blue', 'Marker', 'o', 'LineWidth', 2, 'LineStyle', 'none')
plot(tim(tab.apn > 0), tab.LF(tab.apn > 0), 'Color', 'red', 'Marker', '*', 'LineWidth', 2, 'LineStyle', 'none')
legend('LF', 'Normal', 'Apnea')
xlabel('Time(min)')
ylabel('Amplitude(msec)')
grid on
set(gca, 'fontsize', 15)

subplot(3,1,3)
plot(tim, tab.VLF, 'Color', 'black', 'LineWidth', 1.2)
hold on
plot(tim(tab.apn == 0), tab.VLF(tab.apn == 0), 'Color', 'blue', 'Marker', 'o', 'LineWidth', 2, 'LineStyle', 'none')
plot(tim(tab.apn > 0), tab.VLF(tab.apn > 0), 'Color', 'red', 'Marker', '*', 'LineWidth', 2, 'LineStyle', 'none')
legend('VLF', 'Normal', 'Apnea')
xlabel('Time(min)')
ylabel('Amplitude(msec)')
grid on
set(gca, 'fontsize', 15)

cd plot/
saveas(gca, append(paramName, '.bmp'))
cd ..
end
~~~

~~~Matlab makeTrainData.m
function [tab] = makeTrainData(paramName, Winsize)
cd apnea-ecg-database-1.0.0/
wfdb2mat(paramName);
[tm, data] = rdmat(append(paramName, 'm'));
[tmApn, apn] = rdann(paramName, 'apn');
cd ..

tab = makeLabelofHRV(tm, data, 100, Winsize, tmApn, apn);

for N = 1:height(tab)
    if tab.apn(N) == 0
        tab.label(N) = "Normal";
    else
        tab.label(N) = "Apnea";
    end
end
end
~~~