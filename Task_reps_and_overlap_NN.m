clc;
clear all;
close all;

Orients=2;
Freqs=2;
Rep_per_cond=4; % for example trial, sides, etc. Whatever expands the RDM
% Conditions matrix
Num_conds=Rep_per_cond*Orients*Freqs;
% Labels=Atten_conds*Rep_per_cond*Orients*Freqs
Labels_conds=logical([[zeros(Num_conds./4,1);ones(Num_conds./4,1);zeros(Num_conds./4,1);ones(Num_conds./4,1)],...
    [zeros(Num_conds./8,1);ones(Num_conds./8,1);zeros(Num_conds./8,1);ones(Num_conds./8,1);zeros(Num_conds./8,1);ones(Num_conds./8,1);zeros(Num_conds./8,1);ones(Num_conds./8,1)],...
    [zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1);zeros(Num_conds./16,1);ones(Num_conds./16,1)]]);


NoInput_nodes=4;
NoOuput_nodes=40;














Neurons=40;

Unatt_means=10;
Unatt_variances=3;

Attentional_bias_orient_uni_task=10;
Attentional_bias_orient_mlt_task=0; % will be deducted
Attentional_bias_freq_uni_task=10;
Attentional_bias_freq_mlt_task=5; % will be deducted
Attentiona_vari_orient=5;
Attentiona_vari_freq=5;

plotting=0;

iters=1;
shiftss=0; % 0:5:60
Neurons_per_feature=40;
prp_neurons_coding_ornt=0.5; % proportion of neurons coding orientations
prp_neurons_coding_freq=0.5; % proportion of neurons coding frequencies
prp_neurons_constant_on=0.5; % proportion of neurons remaining on among the non-encoding neurons
mult_tskrs=[0.2]; % 0:0.2:0.8; proportion of multi-tasker neurons: neurons which code both features


shiftee=0;
for shifts=[shiftss]
    shiftee=shiftee+1;
    mt=0;
    for prp_neurons_mult_tskrs=mult_tskrs
        mt=mt+1;
        for iter=[1:iters]
            
            neurons_coding_orient=randsample([1:Neurons_per_feature],Neurons_per_feature*prp_neurons_coding_ornt);
            inds=[1:Neurons_per_feature];
            inds(neurons_coding_orient)=[];
            neurons_on_orient=randsample(inds,Neurons_per_feature*prp_neurons_coding_ornt*prp_neurons_constant_on);
            
            % Assigning similar neurons to the second feature based on the first feature
            inds_similar=randsample(neurons_coding_orient,prp_neurons_mult_tskrs*length(neurons_coding_orient));
            inds=[1:Neurons_per_feature];
            inds(neurons_coding_orient)=[];
            neurons_coding_freq=horzcat(inds_similar,randsample(inds,fix((1-prp_neurons_mult_tskrs)*length(neurons_coding_orient))));
            inds=[1:Neurons_per_feature];
            inds(neurons_coding_freq)=[];
            neurons_on_freq=randsample(inds,Neurons_per_feature*prp_neurons_coding_freq*prp_neurons_constant_on);
            
            
            % putting them in new variables
            Neurons_cod_orient_uni_task=neurons_coding_orient;
            Neurons_cod_orient_mlt_task=inds_similar;
            Neurons_on_orient=neurons_on_orient;
            
            Neurons_cod_freq_uni_task=neurons_coding_freq+shifts;
            Neurons_cod_freq_mlt_task=inds_similar+shifts;
            Neurons_on_freq=neurons_on_freq+shifts;

            %% Generate the unattended base data
            
            Data_unatt=Unatt_means+Unatt_variances.*randn(Neurons,Rep_per_cond*Orients*Freqs);
            
            %% Generate different attended situations where the reps are affected differently by attention
            
            Data_att_orient=Data_unatt;
            Data_att_orient(Neurons_cod_orient_uni_task,Labels_conds(:,2)==1)=Data_att_orient(Neurons_cod_orient_uni_task,Labels_conds(:,2)==1)+Attentional_bias_orient_uni_task+Attentiona_vari_orient*randn(length(Neurons_cod_orient_uni_task),sum(Labels_conds(:,2)==1));
            Data_att_orient(Neurons_cod_orient_mlt_task,Labels_conds(:,2)==1)=Data_att_orient(Neurons_cod_orient_mlt_task,Labels_conds(:,2)==1)+Attentional_bias_orient_mlt_task;
            Data_att_orient(Neurons_on_orient,:)=Data_att_orient(Neurons_on_orient,:)+Attentional_bias_orient_uni_task+Attentiona_vari_orient*randn(length(Neurons_on_orient),size(Labels_conds,1));
            
            Data_att_freq=Data_unatt; % different tasks
            Data_att_freq(Neurons_cod_freq_uni_task,Labels_conds(:,3)==1)=Data_att_freq(Neurons_cod_freq_uni_task,Labels_conds(:,3)==1)+Attentional_bias_freq_uni_task+Attentiona_vari_freq*randn(length(Neurons_cod_freq_uni_task),sum(Labels_conds(:,3)==1));
            Data_att_freq(Neurons_cod_freq_mlt_task,Labels_conds(:,3)==1)=Data_att_freq(Neurons_cod_freq_mlt_task,Labels_conds(:,3)==1)+Attentional_bias_freq_mlt_task;
            Data_att_freq(Neurons_on_freq,:)=Data_att_freq(Neurons_on_freq,:)+Attentional_bias_freq_uni_task+Attentiona_vari_freq*randn(length(Neurons_on_freq),size(Labels_conds,1));
            
            %% Generating neural RDMs
            Data=[Data_att_orient Data_att_freq];

            if plotting==1
                figure;
                imagesc(Data)
            end
            
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
 ccc           
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
            
            
            sz=length(RDM_Conj_Orient_dominant);
            [rColl(shiftee,1,mt,iter),pColl(shiftee,1,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Orient_dominant,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,2,mt,iter),pColl(shiftee,2,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Freq_dominant,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,3,mt,iter),pColl(shiftee,3,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_And,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,4,mt,iter),pColl(shiftee,4,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,5,mt,iter),pColl(shiftee,5,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Or,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,6,mt,iter),pColl(shiftee,6,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_all_stims_across_nan,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,7,mt,iter),pColl(shiftee,7,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_XOR,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,8,mt,iter),pColl(shiftee,8,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_XNOR,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,9,mt,iter),pColl(shiftee,9,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_NOR,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,10,mt,iter),pColl(shiftee,10,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_NAND,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,11,mt,iter),pColl(shiftee,11,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_all_tasks_contrast,[sz*sz 1]),[reshape(RDM_all_stims_across_zeros,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            [rColl(shiftee,12,mt,iter),pColl(shiftee,12,mt,iter)]=partialcorr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_all_stims_across_zeros,[sz*sz 1]),[reshape(RDM_all_tasks_contrast,[sz*sz 1]) reshape(RDM_Null,[sz*sz 1])],'type','pearson','rows','complete');
            if plotting==1
                figure;                
                imagesc(neural_RDM)
                ccc
            end
            sz=length(RDM_Freq_Full);
            [rAcrossTasks(shiftee,1,mt,iter),pAcrossTasks(shiftee,1,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Orient_Full,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,2,mt,iter),pAcrossTasks(shiftee,2,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Freq_Full,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,3,mt,iter),pAcrossTasks(shiftee,3,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_mult,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,4,mt,iter),pAcrossTasks(shiftee,4,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Conj(1:sz,sz+1:end),[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,5,mt,iter),pAcrossTasks(shiftee,5,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_sum,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,6,mt,iter),pAcrossTasks(shiftee,6,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_xor,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,7,mt,iter),pAcrossTasks(shiftee,7,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_xnor,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,8,mt,iter),pAcrossTasks(shiftee,8,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_nor,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            [rAcrossTasks(shiftee,9,mt,iter),pAcrossTasks(shiftee,9,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_nand,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            RDM_rand=randi([0 1],[sz sz]);
            [rAcrossTasks(shiftee,10,mt,iter),pAcrossTasks(shiftee,10,mt,iter)]=partialcorr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_rand,[sz*sz 1]),reshape(RDM_null,[sz*sz 1]),'type','pearson','rows','complete');
            
            [rTaskSpecific(shiftee,1,mt,iter),pTaskSpecific(shiftee,1,mt,iter)]=corr(reshape(neural_orient,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
            [rTaskSpecific(shiftee,2,mt,iter),pTaskSpecific(shiftee,2,mt,iter)]=corr(reshape(neural_orient,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');
            [rTaskSpecific(shiftee,3,mt,iter),pTaskSpecific(shiftee,3,mt,iter)]=corr(reshape(neural_freq,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
            [rTaskSpecific(shiftee,4,mt,iter),pTaskSpecific(shiftee,4,mt,iter)]=corr(reshape(neural_freq,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');
            [rTaskSpecific(shiftee,5,mt,iter),pTaskSpecific(shiftee,5,mt,iter)]=corr(reshape(neural_freq,[sz*sz 1]),reshape(neural_orient,[sz*sz 1]),'type','pearson','rows','complete');
            
            
            [shiftee mt iter]
            
        end
        
    end
end

%% Plotting across shifts
smoothing=1;
mt=1; %level of multi-tasking
figure;
subplot(2,3,1)
for i=[1:12]
    plot(smooth(squeeze(nanmean(rColl(:,i,mt,:),4)),smoothing),'linewidth',3)
    hold on;
end
title('Whole RDM')
xlabel('Difference in neural population (*10%)')
% xlim([0 60])
ylabel('Coding')
legend Orient Freq And Conj Or StimsAlone XOR XNOR NOR NAND Task Stims+zeros
% legend Stimuli Tasks
subplot(2,3,4)
for i=[1:12]
    plot(smooth(squeeze(nanmean(pColl(:,i,mt,:),4)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Difference in neural population (*10%)')
% xlim([0 60])
ylabel('Significance')
legend Orient Freq And Conj Or StimsAlone XOR XNOR NOR NAND Task Stims+zeros
% legend Stimuli Tasks

subplot(2,3,2)
for i=1:10
    plot(smooth(squeeze(nanmean(rAcrossTasks(:,i,mt,:),4)),smoothing),'linewidth',3)
    hold on;
end
title('Across tasks')
xlabel('Difference in neural population (*10%)')
% xlim([0 60])
ylabel('Coding')
legend Orient Freq And Conj Or XOR XNOR NOR NAND RAND
subplot(2,3,5)
for i=1:10
    plot(smooth(squeeze(nanmean(pAcrossTasks(:,i,mt,:),4)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Difference in neural population (*10%)')
% xlim([0 60])
ylabel('Significance')
legend Orient Freq And Conj Or XOR XNOR NOR NAND RAND

subplot(2,3,3)
for i=1:5
    plot(smooth(squeeze(nanmean(rTaskSpecific(:,i,mt,:),4)),smoothing),'linewidth',3)
    hold on;
end
title('Within tasks')
xlabel('Difference in neural population (*10%)')
% xlim([0 60])
ylabel('Coding')
legend Ori-Ori Ori-Frq Frq-Ori Frq-Frq Frq*Ori
subplot(2,3,6)
for i=1:5
    plot(smooth(squeeze(nanmean(pTaskSpecific(:,i,mt,:),4)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Difference in neural population (*10%)')
ylabel('Significance')
legend Ori-Ori Ori-Frq Frq-Ori Frq-Frq Frq*Ori
% xlim([0 60])

%% plotting across neuron's overlap

smoothing=1;
shifts=1; %level of shift in population
figure;
subplot(2,3,1)
for i=[1:12]
    plot(smooth(squeeze(nanmean(rColl(shifts,i,:,:),4)),smoothing),'linewidth',3)
    hold on;
end
title('Whole RDM')
xlabel('Proportion of multi-tasking neurons (*20%)')
% xlim([0 60])
ylabel('Coding')
legend Orient Freq And Conj Or StimsAlone XOR XNOR NOR NAND Task Stims+zeros
% legend Stimuli Tasks
subplot(2,3,4)
for i=[1:12]
    plot(smooth(squeeze(nanmean(pColl(shifts,i,:,:),4)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Proportion of multi-tasking neurons (*20%)')
% xlim([0 60])
ylabel('Significance')
legend Orient Freq And Conj Or StimsAlone XOR XNOR NOR NAND Task Stims+zeros
% legend Stimuli Tasks

subplot(2,3,2)
for i=1:10
    plot(smooth(squeeze(nanmean(rAcrossTasks(shifts,i,:,:),4)),smoothing),'linewidth',3)
    hold on;
end
title('Across tasks')
xlabel('Proportion of multi-tasking neurons (*20%)')
% xlim([0 60])
ylabel('Coding')
legend Orient Freq And Conj Or XOR XNOR NOR NAND RAND
subplot(2,3,5)
for i=1:10
    plot(smooth(squeeze(nanmean(pAcrossTasks(shifts,i,:,:),4)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Proportion of multi-tasking neurons (*20%)')
% xlim([0 60])
ylabel('Significance')
legend Orient Freq And Conj Or XOR XNOR NOR NAND RAND

subplot(2,3,3)
for i=1:5
    plot(smooth(squeeze(nanmean(rTaskSpecific(shifts,i,:,:),4)),smoothing),'linewidth',3)
    hold on;
end
title('Within tasks')
xlabel('Proportion of multi-tasking neurons (*20%)')
% xlim([0 60])
ylabel('Coding')
legend Ori-Ori Ori-Frq Frq-Ori Frq-Frq Frq*Ori
subplot(2,3,6)
for i=1:5
    plot(smooth(squeeze(nanmean(pTaskSpecific(shifts,i,:,:),4)),smoothing),'linewidth',3)
    hold on;
end
xlabel('Proportion of multi-tasking neurons (*20%)')
ylabel('Significance')
legend Ori-Ori Ori-Frq Frq-Ori Frq-Frq Frq*Ori
% xlim([0 60])

%% plotting across neuron's overlap
close all;
figure;
subplot(2,2,1)
surf(squeeze(nanmean(rColl(:,11,:,:),4)))
title('Task')
xlabel('Percentage of multi-tasking neurons (%)')
ylabel('Difference in neural population (%)')
zlabel('Information (\itr)')
set(gca,'xtick',[0:5],'xticklabels',[0:5]*20,'ytick',[0:3:13],'yticklabels',[0:3:13]*10)

subplot(2,2,2)
surf(squeeze(nanmean(rAcrossTasks(:,4,:,:),4)))
title('Conjunction')
xlabel('Percentage of multi-tasking neurons (%)')
ylabel('Difference in neural population (%)')
zlabel('Information (\itr)')
set(gca,'xtick',[0:5],'xticklabels',[0:5]*20,'ytick',[0:3:13],'yticklabels',[0:3:13]*10)

subplot(2,2,3)
surf(squeeze(nanmean(rTaskSpecific(:,1,:,:),4)))
title('Orientation')
xlabel('Percentage of multi-tasking neurons (%)')
ylabel('Difference in neural population (%)')
zlabel('Information (\itr)')
set(gca,'xtick',[0:5],'xticklabels',[0:5]*20,'ytick',[0:3:13],'yticklabels',[0:3:13]*10)


subplot(2,2,4)
surf(squeeze(nanmean(rTaskSpecific(:,4,:,:),4)))
title('Frequency')
xlabel('Percentage of multi-tasking neurons (%)')
ylabel('Difference in neural population (%)')
zlabel('Information (\itr)')
set(gca,'xtick',[0:5],'xticklabels',[0:5]*20,'ytick',[0:3:13],'yticklabels',[0:3:13]*10)
