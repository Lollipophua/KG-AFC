clc;clear;close all;

data = csvread('./data_AMC_4.mat');

% struct = importdata('./ADS-B_10.mat');
% data = getfield(struct, 'data');
% label = getfield(struct, 'label');
% 

data = awgn(data,10);
datafile1 = "/data/huamy/PycharmProjects/pythonProject/pa_noise_10"+".mat"
save(datafile1,'data','data','-v7.3')

[data_feature, data_feature_n] = feature_ex(data);

function [data_feature, data_feature_n] = feature_ex(data)
tempr = data(:,:,1);
tempi = data(:,:,2);
temp = tempr + tempi .* 1j;

fs = 6000;  %采样频率
featureNamesCell = {'max','min','mean','med','peak','arv','var','std','kurtosis',...
                'skewness','rms','rs','rmsa','waveformF','peakF','impulseF','clearanceF',...
               'FC','MSF','RMSF','VF','RVF',...
                'SKMean','SKStd','SKSkewness','SKKurtosis'};   
fea_real = genFeatureTF(tempr,fs,featureNamesCell);
fea_imag = genFeatureTF(tempi,fs,featureNamesCell);

%cum4est
    temp = temp'; 
    data_feature = [];

    for t = 1:36000
        data_cum = temp(:,t);
        M20=mean(data_cum.^2);
        M21=mean(data_cum.*conj(data_cum));
        M40=mean(data_cum.^4);
        M41=mean((data_cum.^3).*conj(data_cum));
        M42=mean((data_cum.^2).*(conj(data_cum).^2));
        M60=mean(data_cum.^6);
        M63=mean((data_cum.^3).*(conj(data_cum).^3));
        C21=abs(M21);
        C40=abs(M40-3*M20^2)/C21^2;
        C41=abs(M41-3*M20*M21)/C21^2;
        C42=abs(M42-abs(M20)^2-2*M21^2)/C21^2;
        C60=abs(M60-15*M40*M20+30*M20^2)/C21^3;
        HOS(:,t) = [C40,C41,C42,C60];
    end
     HOS = HOS';
     data_feature = cat(2,data_feature,fea_real);
     data_feature = cat(2,data_feature,fea_imag);
     data_feature = cat(2,data_feature,HOS);
     data_feature_n = maxmin(data_feature);


     filename = "/data/huamy/PycharmProjects/pythonProject/feature"+".mat"
     save(filename,'data_feature_n','data_feature_n','-v7.3')

end

