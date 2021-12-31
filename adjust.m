function final_results = adjust(S,L,allresults_S1,Train_S,teststress,allresults_L)

for i=1:size(S,1)
    
    if S(i,1)> teststress(i,1)+10
        disp(['第 ',num2str(i),' 个合金应力大于测试应力']);
        adjust = linspace(L(i,1)+5,L(i,1)+500,1000)'; %linspace(下限,上限,次数)
        Com = repmat(allresults_S1(i,1:11),1000,1); %成分列
        T = repmat(allresults_S1(i,12),1000,1); %温度列
        Newset= [Com T adjust];
        [allresults_S,out_S] = model_2 (Train_S,Newset);
        for row = 1:1000
            if  out_S(1,row) < teststress(i,1)+10
                row
                outstress(i,1) = out_S(1,row);
                final_adjust(i,1)= adjust(row,1)
                break
            elseif row == 1000
                disp(['应力预测不准，不调整']);
                outstress(i,1) = S(i,1);
                final_adjust(i,1)= adjust(1,1)-10
                
            end
        end
        
    elseif  S(i,1)<teststress(i,1)-10
        disp(['第 ',num2str(i),' 个合金应力小于测试应力']);
        adjust = linspace(L(i,1)-10,10,100)'; %寿命调整100次，步长为5,寿命列
        Com = repmat(allresults_S1(i,1:11),100,1); %成分列
        T = repmat(allresults_S1(i,12),100,1); %温度列
        Newset= [Com T adjust];
        [allresults_S,out_S] = model_2 (Train_S,Newset);
        for row = 1:100
            if out_S(1,row) > teststress(i,1)-10
                row
                outstress(i,1) = out_S(1,row);
                final_adjust(i,1)= adjust(row,1)
                break
            elseif row==100
                disp(['应力预测不准，不调整']);
                outstress(i,1) = S(i,1);
                final_adjust(i,1)= adjust(1,1)+10
                
            end
        end
        
    else
        disp(['第 ',num2str(i),' 个合金不用调']);
        final_adjust(i,1) = L(i,1)
        outstress(i,1) = S(i,1);
    end
end

final_results = [allresults_L S final_adjust outstress ];%【成分+温度+测试应力+未调寿命 第一次验证应力  调整寿命 最终验证应力】
end