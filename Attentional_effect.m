clc;
clear all;
close all;

Orients=2;
Freqs=2;
Rep_per_cond=4; % for example trial, sides, etc. Whatever expands the RDM
Neurons=100;

Unatt_means=10;
Unatt_variances=3;
Attentional_bias_orient=5;
Attentional_bias_freq=5;
Attentiona_vari_orient=0.1;
Attentiona_vari_freq=0.1;

Neurons_per_feature=50;

xlimL=-5;
xlimH=25;
ylimL=-5;
ylimH=25;

shifts=50; % =0:totally overlapping; >0:non overlapping

neuron_set1_orient=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
neuron_set2_orient=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
neuron_set1_freq=randsample([1:Neurons_per_feature],Neurons_per_feature./2);
neuron_set2_freq=randsample([1:Neurons_per_feature],Neurons_per_feature./2);

Ch_orient1=neuron_set1_orient;
Ch_orient2=neuron_set2_orient;

Ch_freq1=neuron_set1_freq+shifts;
Ch_freq2=neuron_set2_freq+shifts;


scatter_dim1=1;
scatter_dim2=50;
scatter_dim3=99;

%% Generate the unattended base data and visualize it

Data_unatt=Unatt_means+Unatt_variances.*randn(Neurons,Rep_per_cond*Orients*Freqs);
RDM_size=size(Data_unatt,2);
% Labels=Atten_conds*Rep_per_cond*Orients*Freqs
Labels_conds=logical([[zeros(RDM_size./4,1);ones(RDM_size./4,1);zeros(RDM_size./4,1);ones(RDM_size./4,1)],...
    [zeros(RDM_size./8,1);ones(RDM_size./8,1);zeros(RDM_size./8,1);ones(RDM_size./8,1);zeros(RDM_size./8,1);ones(RDM_size./8,1);zeros(RDM_size./8,1);ones(RDM_size./8,1)],...
    [zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1);zeros(RDM_size./16,1);ones(RDM_size./16,1)]]);
%% Generate different attended situations where the reps are affected differently by attention
        Data_att_orient=Data_unatt;
        Data_att_orient(Ch_orient1,Labels_conds(:,2)==1)=Data_att_orient(Ch_orient1,Labels_conds(:,2)==1)+Attentional_bias_orient+Attentiona_vari_orient*randn(length(Ch_orient2),sum(Labels_conds(:,2)==1));
        Data_att_orient(Ch_orient2,Labels_conds(:,2)==0)=Data_att_orient(Ch_orient2,Labels_conds(:,2)==0)+Attentional_bias_orient+Attentiona_vari_orient*randn(length(Ch_orient2),sum(Labels_conds(:,2)==0));
        
        Data_att_freq=Data_unatt;
        Data_att_freq(Ch_freq1,Labels_conds(:,3)==1)=Data_att_freq(Ch_freq1,Labels_conds(:,3)==1)+Attentional_bias_freq+Attentiona_vari_freq*randn(length(Ch_freq1),sum(Labels_conds(:,3)==1));
        Data_att_freq(Ch_freq2,Labels_conds(:,3)==0)=Data_att_freq(Ch_freq2,Labels_conds(:,3)==0)+Attentional_bias_freq+Attentiona_vari_freq*randn(length(Ch_freq2),sum(Labels_conds(:,3)==0));


c=0;
for c=1:RDM_size
    if Labels_conds(c,2)==0 && Labels_conds(c,3)==0
        Colors(c,:)=[0 0 0];
    elseif Labels_conds(c,2)==0 && Labels_conds(c,3)==1
        Colors(c,:)=[1 0 0];
    elseif Labels_conds(c,2)==1 && Labels_conds(c,3)==0
        Colors(c,:)=[0 0 1];
    elseif Labels_conds(c,2)==1 && Labels_conds(c,3)==1
        Colors(c,:)=[1 0 1];
    end
end
Colors=logical(Colors);

sz=30;
subplot(2,2,1)
scatter3(Data_unatt_orient(scatter_dim1,:),Data_unatt_orient(scatter_dim2,:),Data_unatt_orient(scatter_dim3,:),sz,Colors,'filled');
hold on;
line([xlimL xlimH],[0 0])
line([0 0],[ylimL ylimH])
title('Unattended')

subplot(2,2,2)
scatter3(Data_att_orient(scatter_dim1,:),Data_att_orient(scatter_dim2,:),Data_att_orient(scatter_dim3,:),sz,Colors,'filled');
hold on;
line([xlimL xlimH],[0 0])
line([0 0],[ylimL ylimH])
title('Orientation')


subplot(2,2,3)
scatter3(Data_att_freq(scatter_dim1,:),Data_att_freq(scatter_dim2,:),Data_att_freq(scatter_dim3,:),sz,Colors,'filled');
hold on;
line([xlimL xlimH],[0 0])
line([0 0],[ylimL ylimH])
title('Frequency')
%% Generating neural RDMs
Data=[Data_att_orient Data_att_freq];
neural_RDM=nan(size(Data,2));
for n=1:size(Data,2)
    for m=n+1:size(Data,2)
        neural_RDM(n,m)=corr(Data(:,m),Data(:,n));
    end
end
figure
imagesc(Data)
figure;
imagesc(neural_RDM)
caxis([-1 1]);
colorbar;
ccc
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
        neural_conj(n,m)=corr(Data_att_orient(:,m),Data_att_freq(:,n));
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
% figure;
% imagesc(RDM_Both);
% caxis([-1 1]);
% colorbar;

RDM_Orient=nan(size(Data_att_orient,2));
RDM_Freq=nan(size(Data_att_freq,2));
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
    end
end

% figure;
% subplot(1,2,1)
% imagesc(RDM_Orient);
% subplot(1,2,2)
% imagesc(RDM_Freq);
% caxis([-1 1]);
% colorbar;

RDM_Across_Orient=nan(size(Data_att_orient,2));
RDM_Across_Freq=nan(size(Data_att_freq,2));
RDM_Orient_Full=nan(size(Data_att_orient,2));
RDM_Freq_Full=nan(size(Data_att_freq,2));
for n=1:size(Data_att_orient,2)
    for m=1:size(Data_att_orient,2)
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
% figure;
% subplot(1,3,1)
% imagesc(RDM_Across_Orient);
% subplot(1,3,2)
% imagesc(RDM_Across_Freq);
% subplot(1,3,3)
% imagesc(RDM_Across_Conj);
% caxis([-1 1]);
% colorbar;
%% Figures
% sz=16;
% cor_mats(:,:,1)=RDM_Orient_Full;
% cor_mats(:,:,2)=RDM_Freq_Full;
% cor_mats(:,:,3)=RDM_None;
% cor_mats(:,:,4)=RDM_Conj(1:16,17:end);
% for i=1:4
%     for j=1:4
%         [cormat(i,j),pmat(i,j)]=corr(reshape(squeeze(cor_mats(:,:,i)),[sz*sz 1]),reshape(squeeze(cor_mats(:,:,j)),[sz*sz 1]),'type','pearson','rows','complete');
%     end
% end
% subplot(1,2,1);imagesc(cormat);colorbar;subplot(1,2,2);imagesc(pmat);colorbar
% ccc
figure
RDM_Conj_Orient_dominant=[RDM_Orient RDM_Across_Orient;nan(size(RDM_Orient)) RDM_Freq];
RDM_Conj_Freq_dominant=[RDM_Orient RDM_Across_Freq;nan(size(RDM_Orient)) RDM_Freq];
RDM_And=[RDM_Orient RDM_mult;nan(size(RDM_Orient)) RDM_Freq];
RDM_Or=[RDM_Orient RDM_sum;nan(size(RDM_Orient)) RDM_Freq];
sz=length(RDM_Conj_Orient_dominant);

subplot(2,3,1)
imagesc(RDM_Conj_Orient_dominant)
[r,p]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Orient_dominant,[sz*sz 1]),'type','pearson','rows','complete');
title(['Orientation r=',num2str(r),'; p=',num2str(p)])
subplot(2,3,2)
imagesc(RDM_Conj_Freq_dominant)
[r,p]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj_Freq_dominant,[sz*sz 1]),'type','pearson','rows','complete');
title(['Freq r=',num2str(r),'; p=',num2str(p)])
subplot(2,3,3)
imagesc(RDM_And)
[r,p]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_And,[sz*sz 1]),'type','pearson','rows','complete');
title(['And r=',num2str(r),'; p=',num2str(p)])
subplot(2,3,4)
imagesc(RDM_Conj);
[r,p]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Conj,[sz*sz 1]),'type','pearson','rows','complete');
title(['Conj r=',num2str(r),'; p=',num2str(p)])
subplot(2,3,5)
imagesc(RDM_Or);
[r,p]=corr(reshape(neural_RDM,[sz*sz 1]),reshape(RDM_Or,[sz*sz 1]),'type','pearson','rows','complete');
title(['Or r=',num2str(r),'; p=',num2str(p)])


sz=length(RDM_Orient);
[rOO,pOO]=corr(reshape(neural_orient,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
[rOF,pOF]=corr(reshape(neural_orient,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');
[rFO,pFO]=corr(reshape(neural_freq,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
[rFF,pFF]=corr(reshape(neural_freq,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');
[rCO,pCO]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Orient,[sz*sz 1]),'type','pearson','rows','complete');
[rCF,pCF]=corr(reshape(neural_conj,[sz*sz 1]),reshape(RDM_Freq,[sz*sz 1]),'type','pearson','rows','complete');

[rOO rOF rFO rFF rCO rCF;pOO pOF pFO pFF pCO pCF]

[rCO rCF;pCO pCF]

