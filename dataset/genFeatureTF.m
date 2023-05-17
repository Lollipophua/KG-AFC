function fea = genFeatureTF(data,fs,featureNamesCell)
rng('default')
data = data';
[len,num] = size(data);
if len == 1
    data = data';
    [len,num] = size(data);
end
allFeaNames = {'max','min','mean','med','peak','arv','var','std','kurtosis',...
               'skewness','rms','rs','rmsa','waveformF','peakF','impulseF','clearanceF',...
               'FC','MSF','RMSF','VF','RVF',...
               'SKMean','SKStd','SKSkewness','SKKurtosis'};
%% 1.time
ma = max(data);
mi = min(data);	
me = mean(data); 
med = median(data);
pk = ma-mi;	
av = mean(abs(data));
va = var(data);	
st = std(data);	
ku = kurtosis(data);
sk = skewness(data); 
rm = rms(data);
rs = rm.^2; 
rmsa = mean(sqrt(abs(data)),1).^2;
S = rm./av;	
C = pk./rm;  
I = pk./av;  
L = pk./mean(sqrt(abs(data))).^2;

allTimeFea = [ma;mi;me;med;pk;av;va;st;ku;sk;rm;rs;rmsa;S;C;I;L];
%% 2.frequency
FC = zeros(1,num);MSF=zeros(1,num);RMSF=zeros(1,num);VF=zeros(1,num);RVF=zeros(1,num); 
if sum(contains(featureNamesCell,{'FC','MSF','RMSF','VF','RVF'}))>0 
    [p,f] = periodogram(data,[],[],fs); 
    FC = sum(p.*f)./sum(p);
    MSF = sum(f.^2.*p)./sum(p);
    RMSF = sqrt(MSF);
    VF = sum((f-FC).^2.*p)./sum(p);
    RVF = sqrt(VF);
end
allTimeFea = [allTimeFea;FC;MSF;RMSF;VF;RVF]; 

%% 3.specture
SK = spectralKurtosis(data,fs); 
SKMean = mean(SK);
SKStd = std(SK);
SKSkewness = skewness(SK);
SKKurtosis = kurtosis(SK);
allFea = [allTimeFea;SKMean;SKStd;SKSkewness;SKKurtosis];

fea = [];
for i = 1:length(featureNamesCell)
    try
    if find(contains(allFeaNames,featureNamesCell{i}))
        fea = [fea;allFea(find(strcmp(allFeaNames,featureNamesCell{i})),:)];
    end
    catch ME 
    end
end

fea = fea';

end


