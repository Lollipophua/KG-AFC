clc;clear ;close all;
L = 6000; % 单个样本数目
m = 12000; % 单个样本读取长度
s = m/2;  % IQ分别为多少
num_RFF = 6;  % 设备序号
num_AMC = 8;  % 调制类型序号
t = 4;
data = zeros(L*num_RFF,s,2);
RFF_label = zeros(L*num_RFF,1);
for r=1:num_RFF
    % 解析代码跟数据放在同一个文件夹下就行了。这里就不用修改了，不在一个文件夹中修改路径
    fileName = strcat('RFF_',num2str(r),'_AMC_',num2str(t),'.iq'); % 读取文件名称
    fid = fopen(fileName, 'r');
    IQ1 = fread(fid, [1,L*m], 'int16');
    status = fclose(fid);
    data_max = max(abs(IQ1));
    IQ2 = IQ1/data_max;
    %         IQ3 = awgn(IQ2, 15,'measured');  % 对IQ2 加噪声，信噪比为15
    for j = 1:L
        data(((r-1)*L+j),1:s,1)=IQ2(1,(((j-1)*m)+1):2:((j)*m));
        data(((r-1)*L+j),1:s,2)=IQ2(1,(((j-1)*m)+2):2:((j)*m));
        RFF_label(((r-1)*L+j),1) = (r-1);
    end
end
clear IQ1 IQ2 fid IQ3

% 这儿
% path_name = 'E:\RE_RFF_AMC\data\20dB\';


% 这儿
file_name = strcat('data_AMC_',num2str(t),'.mat');
save (file_name,'data','RFF_label','-v7.3')




