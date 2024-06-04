clear
load('MFF_DenseNet.mat')
load('noisydata_sample.mat')

%If want to test your own data, please use 200×1×1 input
%This example was designed to test the internal dynamics of MFF-DenseNet 
%in the face of noise and signal.
%% feature maps visulization
if canUseGPU
    X = gpuArray(A1');
end
fd=200;
Layer11 = "concat_2";
Denseblock1Features = activations(MFF_DenseNet,X,Layer11);
% Calculate the average activation value for each feature map
D1avg_activations = mean(mean(Denseblock1Features, 1), 2);
D1avg_activations = squeeze(D1avg_activations);
% Sort the feature maps by their average activation value and select the top 64
[D1sorted_activations, D1sorted_indices] = sort(D1avg_activations, 'descend');
D1top_feature_maps = Denseblock1Features(:, :, D1sorted_indices(1:64));
D1top_feature_maps=squeeze(D1top_feature_maps);

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(D1top_feature_maps(:, i));%13   
end
title("Dense block extracted feature maps");

Layer1 = "concat_8";
Denseblock2Features = activations(MFF_DenseNet,X,Layer1);
% Calculate the average activation value for each feature map
D2avg_activations = mean(mean(Denseblock2Features, 1), 2);
D2avg_activations = squeeze(D2avg_activations);
% Sort the feature maps by their average activation value and select the top 64
[D2sorted_activations, D2sorted_indices] = sort(D2avg_activations, 'descend');
D2top_feature_maps = Denseblock2Features(:, :, D2sorted_indices(1:64));
D2top_feature_maps=squeeze(D2top_feature_maps);
%plot feature maps extracted by Dense block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(D2top_feature_maps(:, i));%13   
end
title("Dense block extracted feature maps");

Layer2 = "concat_10_1";
CSLFeatures = activations(MFF_DenseNet,X,Layer2);
CSLavg_activations = mean(mean(CSLFeatures, 1), 2);
CSLavg_activations = squeeze(CSLavg_activations);
[CSLsorted_activations, CSLsorted_indices] = sort(CSLavg_activations, 'descend');
CSLtop_feature_maps = CSLFeatures(:, :, CSLsorted_indices(1:64));
CSLtop_feature_maps=squeeze(CSLtop_feature_maps);

%plot feature maps extracted by CSL module

figure;
num_maps = 64;
for i = 1:num_maps
    % Create a subplot for the activation map in a 5x5 grid
    subplot(8, 8, i);
   plot(CSLtop_feature_maps(:, i));
    
end
title( "CSL module extracted feature maps");
%% SPP Features
Layer3 = "concat_10";
SPPFeatures = activations(MFF_DenseNet,X,Layer3);
SPPavg_activations = mean(mean(SPPFeatures, 1), 2);
SPPavg_activations = squeeze(SPPavg_activations);
[SPPsorted_activations, SPPsorted_indices] = sort(SPPavg_activations, 'descend');
SPPtop_feature_maps = SPPFeatures(:, :, SPPsorted_indices(1:64));
SPPtop_feature_maps=squeeze(SPPtop_feature_maps);

%plot feature maps extracted by SPP module

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(SPPtop_feature_maps(:, i));
    
end
title( "SPP extracted feature maps");


%%
MT=A1;Dnoise=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        Dnoise=[Dnoise,a-DEX'];
    end
end
CONTOUR=A1-Dnoise;
%% SIG
load('signal_sample.mat')
if canUseGPU
    X2 = gpuArray(A2');
end

Layer11 = "concat_2";
SIGDenseblock1Features = activations(MFF_DenseNet,X2,Layer11);
% Calculate the average activation value for each feature map
SIGD1avg_activations = mean(mean(SIGDenseblock1Features, 1), 2);
SIGD1avg_activations = squeeze(SIGD1avg_activations);
% Sort the feature maps by their average activation value and select the top 64
[SIGD1sorted_activations, SIGD1sorted_indices] = sort(SIGD1avg_activations, 'descend');
SIGD1top_feature_maps = SIGDenseblock1Features(:, :, SIGD1sorted_indices(1:64));
SIGD1top_feature_maps=squeeze(SIGD1top_feature_maps);

Layer1 = "concat_8";
SIGDenseblock2Features = activations(MFF_DenseNet,X2,Layer1);
SIGD2avg_activations = mean(mean(SIGDenseblock2Features, 1), 2);
SIGD2avg_activations = squeeze(SIGD2avg_activations);
[SIGD2sorted_activations, SIGD2sorted_indices] = sort(SIGD2avg_activations, 'descend');
SIGD2top_feature_maps = SIGDenseblock2Features(:, :, SIGD2sorted_indices(1:64));
SIGD2top_feature_maps=squeeze(SIGD2top_feature_maps);

%plot feature maps extracted by Dense Block 1
figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(SIGD2top_feature_maps(:, i));
    
end

%plot feature maps extracted by Dense Block 2
figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(SIGD1top_feature_maps(:, i));
    
end
title("Dense block extracted feature maps");

Layer2 = "concat_10_1";
SIGCSLFeatures = activations(MFF_DenseNet,X2,Layer2);
SIGCSLavg_activations = mean(mean(SIGCSLFeatures, 1), 2);
SIGCSLavg_activations = squeeze(SIGCSLavg_activations);
[SIGCSLsorted_activations, SIGCSLsorted_indices] = sort(SIGCSLavg_activations, 'descend');
SIGCSLtop_feature_maps = SIGCSLFeatures(:, :, SIGCSLsorted_indices(1:64));
SIGCSLtop_feature_maps=squeeze(SIGCSLtop_feature_maps);

%plot feature maps extracted by CSL moduel
figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(SIGCSLtop_feature_maps(:, i));
    
end
title( "CSL module extracted feature maps");
%% SPP Features
Layer3 = "concat_10";
SIGSPPFeatures = activations(MFF_DenseNet,X2,Layer3);
SIGSPPavg_activations = mean(mean(SIGSPPFeatures, 1), 2);
SIGSPPavg_activations = squeeze(SIGSPPavg_activations);
[SIGSPPsorted_activations, SIGSPPsorted_indices] = sort(SIGSPPavg_activations, 'descend');
SIGSPPtop_feature_maps = SIGSPPFeatures(:, :, SIGSPPsorted_indices(1:64));
SIGSPPtop_feature_maps=squeeze(SIGSPPtop_feature_maps);
%plot feature maps extracted by SPP moduel
figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(SIGSPPtop_feature_maps(:, i));
    
end
title( "SPP extracted feature maps");

%% Denoising
MT=A2;Dnoise2=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        Dnoise2=[Dnoise2,a-DEX'];
    end
end
CONTOUR2=A2-Dnoise2;
%%
figure
subplot(3, 4, 1)
plot(A1,Color='k'),title("Noisy MT data");text('Units', 'normalized', 'String', '(a)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 5)
plot(D1top_feature_maps(:, 39),Color='k'),title("Dense Block");text('Units', 'normalized', 'String', '(b)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 9)
plot(D2top_feature_maps(:, 20),Color='k'),title("Dense Block");text('Units', 'normalized', 'String', '(c)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 2)
plot(CSLtop_feature_maps(:, 1),Color='k'),title("CSL module");text('Units', 'normalized', 'String', '(d)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 6)
plot(SPPtop_feature_maps(:, 3),Color='k'),title("SPP module");text('Units', 'normalized', 'String', '(e)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 10)
plot(CONTOUR,Color='k')
hold on
plot(Dnoise,Color='r');title("Output Noise");text('Units', 'normalized', 'String', '(f)', 'Position',[0.01 0.93 0], 'FontSize', 12);
legend('Network output','Reconstructed signal')
subplot(3, 4, 3)
plot(A2,Color='k'),title("High-quality MT data");text('Units', 'normalized', 'String', '(g)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 7)
plot(SIGD1top_feature_maps(:, 29),Color='k'),title("Dense Block");text('Units', 'normalized', 'String', '(h)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 11)
plot(SIGD2top_feature_maps(:, 59),Color='k'),title("Dense Block");text('Units', 'normalized', 'String', '(i)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 4)
plot(SIGCSLtop_feature_maps(:, 24),Color='k'),title("CSL module");text('Units', 'normalized', 'String', '(j)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 8)
plot(SIGSPPtop_feature_maps(:, 28),Color='k'),title("SPP module");text('Units', 'normalized', 'String', '(k)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(3, 4, 12)
plot(CONTOUR2,Color='k')
hold on
plot(Dnoise2,Color='r');title("Output Signal");text('Units', 'normalized', 'String', '(l)', 'Position',[0.01 0.93 0], 'FontSize', 12);
legend('Network output','Reconstructer signal')


