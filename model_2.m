function [allresults_S,out_S] = model_2 (Train_S,Test_S)
%A_S = [Train_S' Test_S'];
%[A_S,ps]=mapminmax(A_S);
[normTrain_S,ps]=mapminmax(Train_S');
normtestinput_S = mapminmax('apply',Test_S',ps);

% 训练输入数据导入
normInset_S = normTrain_S;%不用动

% 训练的真实输出导入及整理
t = xlsread('C:\Users\Uaena_HY\Desktop\代码集\数据库\newset_stress','153','O2:O154');
t=t';
[t,ts]=mapminmax(t); %输出数据归一化
% 训练的输出数据导入
normOutput_S=t;%不用动


for s=1:15  %预测50次，取平均值，模拟真实的预测情况时使用
    % 数据乱序
    RD=randperm(size(Train_S',2));% 95%的数据
    normInput_S=normInset_S(:,RD(1:145));
    normTarget_S=normOutput_S(1,RD(1:145));
    %%创建网络
    net_S=newff(minmax(normInput_S),[9,1],{'tansig','purelin'},'traingdx');
    %%权值和阈值
    inputWeights=net_S.iW{1,1};
    inputbias=net_S.b{1};
    layerWeights=net_S.lW{2,1};
    layerbias=net_S.b{2};
    
    %%网络参数设置
    net_S.trainParam.epochs=1000;
    net_S.trainParam.goal=1e-4;
    net_S.trainParam.lr=0.1;
    net_S.trainParam.show = 50;
       
    %%训练网络
    [ net_S, tr] = train( net_S,normInput_S,normTarget_S);
    
    
    %%仿真
    
    %训练数据仿真
    normtrainoutput=sim(net_S,normInput_S);
    
    %测试数据仿真
    %number=180+size(Test_S,1);  %测试数据个数
    %normtestinput_S=A_S(:,(181:number)); %测试数据
    normtestoutput_S=sim(net_S,normtestinput_S);  %仿真
    %测试数据反归一化
    testinput_S=mapminmax('reverse',normtestinput_S,ps);
    %%结果反归一化
    trainOutput_S=mapminmax('reverse',normtrainoutput,ts); %训练数据仿真结果
    testOutput_S=mapminmax('reverse',normtestoutput_S,ts); %测试数据的预测结果
    
    preout_S(s,:) = abs(testOutput_S);
    
end
out_S=mean(preout_S);
allresults_S=[testinput_S' out_S'];
end
