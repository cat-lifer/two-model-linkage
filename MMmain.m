close all; clear; clc; warning('off');
%% 模块1 寿命预测
% 训练和预测的输入数据导入和归一化
Train_L=xlsread('C:\Users\Uaena_HY\Desktop\代码集\数据库\153','Sheet3','B2:N154');
%Train_L=xlsread('C:\Users\DELL\Desktop\Test\总表','More TCP','B2:N65');%tcp
Test_L = xlsread('C:\Users\Uaena_HY\Desktop\代码集\Testset\预备熔炼','Sheet1','B4:N6'); 
[allresults_L,out_L] = model_1 (Train_L,Test_L);

%% 模块2 应力预测
Train_S=xlsread('C:\Users\Uaena_HY\Desktop\代码集\数据库\newset_stress','153','B2:N154');
Test_S = allresults_L(:,[1:12 14]); %成分+温度 寿命 
[allresults_S,out_S] = model_2(Train_S,Test_S);
allresults_S1=allresults_S; %成分 温度 预测寿命 预测应力
out_S1=out_S;

%% 模块3 寿命调整
S=out_S1';
L=out_L';
teststress = allresults_L(:,13);
final_results = adjust(S,L,allresults_S1,Train_S,teststress,allresults_L);%【成分+温度+测试应力+未调寿命 第一次验证应力  调整寿命 最终验证应力】