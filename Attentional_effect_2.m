clc;
clear all;
close all;

Orients=2;
Freqs=2;
Rep_per_cond=4; % for example trial, sides, etc. Whatever expands the RDM
Neurons=100;

Unatt_means=10;
Unatt_variances=3;

Attentional_bias_orient=50;
Attentional_bias_freq=50;
Attentiona_vari_orient=1;
Attentiona_vari_freq=1;

Neurons_per_feature=40;
iters=50;

for shifts=[0:60]
    shiftee=shifts+1;
    for iter=[1:iters]
        neuron_set1_orient=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
        neuron_set2_orient=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
        neuron_set1_freq=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
        neuron_set2_freq=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
        
        Ch_orient1=neuron_set1_orient;
        Ch_orient2=neuron_set2_orient;
        
%         Ch_freq1=neuron_set1_freq+shifts; % diffeent populations across tasks
%         Ch_freq2=neuron_set2_freq+shifts;

        Ch_freq1=neuron_set1_orient+shifts; % similar populations across tasks
        Ch_freq2=neuron_set2_orient+shifts;
     
        %% Generate the unattended base data and visualize it
        
        Data_unatt=Unatt_means+Unatt_variances.*randn(Neurons,Rep_per_cond*Orients*Freqs);
        RDM_size=size(Data_unatt,2);
        % Labels=Atten_conds*Rep_per_cond*Orients*Freqs
        Labels_conds=logical([[zeros(RDM_size./4,1);ones(RDM_size./4,1);zeros(RDM_size./4,1);ones(RDM_size./4,1)],...
            [zeros(RDM_size./8,1);ones(RDM_size./8,1);zeros(RDM_size./8,1);ones(RDM_size./8,1);zeros(RDM_size./8,1);ones(RDM_size./8,1);zeros(RDM_size./8,1);ones(RDM_size./8,1)],...
            [zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1)]]);
        
        
        %% Generate different attended situations where the reps are affected differently by attention
        
        Data_att_orient=Data_unatt;
        Data_att_orient(Ch_orient1,Labels_conds(:,2)==1)=Data_att_orient(Ch_orient1,Labels_conds(:,2)==1)+Attentional_bias_orient+Attentiona_vari_orient*randn(length(Ch_orient1),sum(Labels_conds(:,2)==1));
        Data_att_orient(Ch_orient2,Labels_conds(:,2)==0)=Data_att_orient(Ch_orient2,Labels_conds(:,2)==0)+Attentional_bias_orient+Attentiona_vari_orient*randn(length(Ch_orient2),sum(Labels_conds(:,2)==0));
        
%         Data_att_freq=Data_unatt; % different tasks
%         Data_att_freq(Ch_freq1,Labels_conds(:,3)==1)=Data_att_freq(Ch_freq1,Labels_conds(:,3)==1)+Attentional_bias_freq+Attentiona_vari_freq*randn(length(Ch_freq1),sum(Labels_conds(:,3)==1));
%         Data_att_freq(Ch_freq2,Labels_conds(:,3)==0)=Data_att_freq(Ch_freq2,Labels_conds(:,3)==0)+Attentional_bias_freq+Attentiona_vari_freq*randn(length(Ch_freq2),sum(Labels_conds(:,3)==0));
        
        Data_att_freq=Data_unatt;   % similar tasks
        Data_att_freq(Ch_freq1,Labels_conds(:,2)==1)=Data_att_freq(Ch_freq1,Labels_conds(:,2)==1)+Attentional_bias_freq+Attentiona_vari_freq*randn(length(Ch_freq1),sum(Labels_conds(:,2)==1));
        Data_att_freq(Ch_freq2,Labels_conds(:,2)==0)=Data_att_freq(Ch_freq2,Labels_conds(:,2)==0)+Attentional_bias_freq+Attentiona_vari_freq*randn(length(Ch_freq2),sum(Labels_conds(:,2)==0));
       
        %% Generating neural RDMs
        Data=[Data_att_orient Data_att_freq];
        neural_RDM=nan(size(Data,2));
        for n=1:size(Data,2)
            for m=n+1:size(Data,2)
                neural_RDM(n,m)=corr(Data(:,m),Data(:,n));
            end
        end
        
        % Generating neural sub-RDMs
        neural_orient=nan(size(Data_att_orient,2));
        neural_freq=nan(size(Data_att_freq,2));
        
        for n=1:size(Data_att_orient,2)
            for m=n+1:size(Data_att_freq,2)
                neural_orient(n,m)=corr(Data_att_orient(:,m),Data_att_orient(:,n));
                neural_freq(n,m)=corr(Data_att_freq(:,m),Data_att_freq(:,n));
            end
        end
        
        neural_conj=nan(size(Data_att_freq,2));
        for n=1:size(Data_att_orient,2)
            for m=1:size(Data_att_freq,2)
                neural_conj(m,n)=corr(Data_att_orient(:,m),Data_att_freq(:,n));
            end
        end
        
        %% Generating model RDMs
        RDM_Conj=nan(size(Data,2));
        Labels_Total=[Labels_conds(:,2);Labels_conds(:,3)];
        for n=1:size(Data,2)
            for m=n+1:size(Data,2)
                if Labels_Total(n)==Labels_Total(m)
                    RDM_Conj(n,m)=1;
                else
                    RDM_Conj(n,m)=0;
                end
            end
        end
        
        RDM_Orient=nan(size(Data_att_orient,2));
        RDM_Freq=nan(size(Data_att_freq,2));
        RDM_zero=nan(size(Data_att_freq,2));
        RDM_ones=nan(size(Data_att_freq,2));
        for n=1:size(Data_att_orient,2)
            for m=n+1:size(Data_att_orient,2)
                if Labels_conds(n,2)==Labels_conds(m,2)
                    RDM_Orient(n,m)=1;
                else
                    RDM_Orient(n,m)=0;
                end
                if Labels_conds(n,3)==Labels_conds(m,3)
                    RDM_Freq(n,m)=1;
                else
                    RDM_Freq(n,m)=0;
                end
                RDM_zero(n,m)=0;
                RDM_ones(n,m)=1;
            end
        end
        
        RDM_Across_Orient=nan(size(Data_att_orient,2));
        RDM_Across_Freq=nan(size(Data_att_freq,2));
        RDM_Orient_Full=nan(size(Data_att_orient,2));
        RDM_Freq_Full=nan(size(Data_att_freq,2));
        RDM_null=nan(size(Data_att_freq,2));
        for n=1:size(Data_att_orient,2)
            for m=1:size(Data_att_orient,2)
                if m==n
                    RDM_null(n,m)=1;
                else
                    RDM_null(n,m)=0;
                end
                if Labels_conds(n,2)==Labels_conds(m,2)
                    RDM_Across_Orient(n,m)=1;
                else
                    RDM_Across_Orient(n,m)=0;
                end
                if Labels_conds(n,3)==Labels_conds(m,3)
                    RDM_Across_Freq(n,m)=1;
                else
                    RDM_Across_Freq(n,m)=0;
                end
                if Labels_conds(n,2)==Labels_conds(m,2)
                    RDM_Orient_Full(n,m)=1;
                else
                    RDM_Orient_Full(n,m)=0;
                end
                if Labels_conds(n,3)==Labels_conds(m,3)
                    RDM_Freq_Full(n,m)=1;
                else
                    RDM_Freq_Full(n,m)=0;
                end
            end
        end
        RDM_mult=RDM_Across_Freq.*RDM_Across_Orient;
        RDM_sum=double(RDM_Across_Freq | RDM_Across_Orient);
        
        RDM_nor=double(RDM_Across_Freq | RDM_Across_Orient);
        RDM_nand=double(~(RDM_Across_Freq & RDM_Across_Orient));
        RDM_xor=double((xor(RDM_Across_Freq,RDM_Across_Orient)));
        RDM_xnor=double(~(xor(RDM_Across_Freq,RDM_Across_Orient)));
        
        %% Correlations
        RDM_Conj_Orient_dominant=[RDM_Orient RDM_Across_Orient;nan(size(RDM_Orient)) RDM_Freq];
        RDM_Conj_Freq_dominant=[RDM_Orient RDM_Across_Freq;nan(size(RDM_Orient)) RDM_Freq];
        RDM_And=[RDM_Orient RDM_mult;nan(size(RDM_Orient)) RDM_Freq];
        RDM_Or=[RDM_Orient RDM_sum;nan(size(RDM_Orient)) RDM_Freq];
        RDM_Null=[RDM_zero RDM_null;nan(size(RDM_Orient)) RDM_zero];
        
        RDM_all_stims_across_nan=[RDM_Orient nan(size(RDM_Orient));nan(size(RDM_Orient)) RDM_Freq];
        RDM_all_tasks_contrast=[RDM_ones zeros(size(RDM_Orient));nan(size(RDM_Orient)) RDM_ones];
        RDM_all_stims_across_zeros=[RDM_Orient zeros(size(RDM_Orient));nan(size(RDM_Orient)) RDM_Freq];
        RDM_all_stims_across_ones=[RDM_Orient ones(size(RDM_Orient));nan(size(RDM_Orient)) RDM_Freq];
        
        RDM_XOR=[RDM_Orient RDM_xor;nan(size(RDM_Orient)) RDM_Freq];
        RDM_XNOR=[RDM_Orient RDM_xnor;nan(size(RDM_Orient)) RDM_Freq];
        RDM_NAND=[RDM_Orient RDM_nand;nan(size(RDM_Orient)) RDM_Freq];
        RDM_NOR=[RDM_Orient RDM_nor;nan(size(RDM_Orient)) RDM_Freq];

        %         sz=length(RDM_Conj_Orient_dominant);
        %         [rColl(shiftee,1,iter),pColl(shiftee,1,iter)]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Orient_dominant,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rColl(shiftee,2,iter),pColl(shiftee,2,iter)]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Freq_dominant,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rColl(shiftee,3,iter),pColl(shiftee,3,iter)]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_And,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rColl(shiftee,4,iter),pColl(shiftee,4,iter)]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rColl(shiftee,5,iter),pColl(shiftee,5,iter)]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Or,[sz*sz 1]),'type','pearson','rows','complete');
        %
        %         sz=length(RDM_Freq_Full);
        %         [rAcrossTasks(shiftee,1,iter),pAcrossTasks(shiftee,1,iter)]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Orient_Full,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rAcrossTasks(shiftee,2,iter),pAcrossTasks(shiftee,2,iter)]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Freq_Full,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rAcrossTasks(shiftee,3,iter),pAcrossTasks(shiftee,3,iter)]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_mult,[sz*sz 1]),'type','pearson','rows','complete');
        %         [rAcrossTasks(shiftee,4,iter),pAcrossTasks(shiftee,4,iter)]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Conj(1:sz,sz+1:end),[sz*sz 1]),'type','pearson','rows','complete');
        %         [rAcrossTasks(shiftee,5,iter),pAcrossTasks(shiftee,5,iter)]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_sum,[sz*sz 1]),'type','pearson','rows','complete');
        
        
        sz=length(RDM_Conj_Orient_dominant);
        [rColl(shiftee,1,iter),pColl(shiftee,1,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Orient_dominant,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,2,iter),pColl(shiftee,2,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Freq_dominant,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,3,iter),pColl(shiftee,3,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_And,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,4,iter),pColl(shiftee,4,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,5,iter),pColl(shiftee,5,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Or,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,6,iter),pColl(shiftee,6,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_all_stims_across_nan,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,7,iter),pColl(shiftee,7,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_XOR,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,8,iter),pColl(shiftee,8,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_XNOR,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,9,iter),pColl(shiftee,9,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_NOR,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,10,iter),pColl(shiftee,10,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_NAND,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,11,iter),pColl(shiftee,11,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_all_tasks_contrast,[sz*sz 1]),[reshape(RDM_all_stims_across_zeros,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
        [rColl(shiftee,12,iter),pColl(shiftee,12,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_all_stims_across_zeros,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');

        sz=length(RDM_Freq_Full);
        [rAcrossTasks(shiftee,1,iter),pAcrossTasks(shiftee,1,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Orient_Full,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,2,iter),pAcrossTasks(shiftee,2,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Freq_Full,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,3,iter),pAcrossTasks(shiftee,3,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_mult,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,4,iter),pAcrossTasks(shiftee,4,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Conj(1:sz,sz+1:end),[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,5,iter),pAcrossTasks(shiftee,5,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_sum,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,6,iter),pAcrossTasks(shiftee,6,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_xor,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,7,iter),pAcrossTasks(shiftee,7,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_xnor,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,8,iter),pAcrossTasks(shiftee,8,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_nor,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        [rAcrossTasks(shiftee,9,iter),pAcrossTasks(shiftee,9,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_nand,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        RDM_rand=randi([0 1],[sz sz]);
        [rAcrossTasks(shiftee,10,iter),pAcrossTasks(shiftee,10,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_rand,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
        
        [rTaskSpecific(shiftee,1,iter),pTaskSpecific(shiftee,1,iter)]=corr(reshape(neural_orient,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
        [rTaskSpecific(shiftee,2,iter),pTaskSpecific(shiftee,2,iter)]=corr(reshape(neural_orient,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');
        [rTaskSpecific(shiftee,3,iter),pTaskSpecific(shiftee,3,iter)]=corr(reshape(neural_freq,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
        [rTaskSpecific(shiftee,4,iter),pTaskSpecific(shiftee,4,iter)]=corr(reshape(neural_freq,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');
        [rTaskSpecific(shiftee,5,iter),pTaskSpecific(shiftee,5,iter)]=corr(reshape(neural_freq,[sz*sz 1]),reshape(neural_orient,[sz*sz 1]),'type','pearson','rows','complete');
        
        
        [shiftee iter]
        
    end
end
%% Plotting across shifts
smoothing=1;
subplot(2,3,1)
for i=[1:12]
    plot(smooth(squeeze(nanmean(rColl(:,i,:),3)),smoothing),'linewidth',3)
    hold on;
end
title('Whole RDM')
xlabel('Difference in neural population')
xlim([0 60])
ylabel('Coding')
legend Orient Freq And Conj Or StimsAlone XOR XNOR NOR NAND Task Stims+zeros
% legend Stimuli Tasks
subplot(2,3,4)
for i=[1:12]
    plot(smooth(squeeze(nanmean(pColl(:,i,:),3)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Difference in neural population')
xlim([0 60])
ylabel('Significance')
legend Orient Freq And Conj Or StimsAlone XOR XNOR NOR NAND Task Stims+zeros
% legend Stimuli Tasks

subplot(2,3,2)
for i=1:10
    plot(smooth(squeeze(nanmean(rAcrossTasks(:,i,:),3)),smoothing),'linewidth',3)
    hold on;
end
title('Across tasks')
xlabel('Difference in neural population')
xlim([0 60])
ylabel('Coding')
legend Orient Freq And Conj Or XOR XNOR NOR NAND RAND
subplot(2,3,5)
for i=1:10
    plot(smooth(squeeze(nanmean(pAcrossTasks(:,i,:),3)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Difference in neural population')
xlim([0 60])
ylabel('Significance')
legend Orient Freq And Conj Or XOR XNOR NOR NAND RAND

subplot(2,3,3)
for i=1:5
    plot(smooth(squeeze(nanmean(rTaskSpecific(:,i,:),3)),smoothing),'linewidth',3)
    hold on;
end
title('Within tasks')
xlabel('Difference in neural population')
xlim([0 60])
ylabel('Coding')
legend Ori-Ori Ori-Frq Frq-Ori Frq-Frq
subplot(2,3,6)
for i=1:5
    plot(smooth(squeeze(nanmean(pTaskSpecific(:,i,:),3)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Difference in neural population')
ylabel('Significance')
legend Ori-Ori Ori-Frq Frq-Ori Frq-Frq
xlim([0 60])