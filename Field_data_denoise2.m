clear
%% load data and model
A=load('Field_dataA.dat');
load('Trained_without_2modules.mat')
load('Trained_without_SPP.mat')
load('Trained_without_CSL.mat')
load('MFF_DenseNet.mat')
%% Denoising
A=A-mean(A);
MT=A(15001:19000);fd=200;
DenseDenoise=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(DenseNet,In);
        DEX=reshape(out,fd,1);
        DenseDenoise=[DenseDenoise,a-DEX'];
    end
end

CSLDenoise=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(DenseNet_CSL,In);
        DEX=reshape(out,fd,1);
        CSLDenoise=[CSLDenoise,a-DEX'];
    end
end
SPPDenoise=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(DenseNet_SPP,In);
        DEX=reshape(out,fd,1);
        SPPDenoise=[SPPDenoise,a-DEX'];
    end
end
Denoise=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        Denoise=[Denoise,a-DEX'];
    end
end
%% Figure 20
figure()
subplot(3,2,1),plot(MT,'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(DenseDenoise,'g'),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(a)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('MFF-DenseNet(Without CSL and SPP)');axis("auto xy")
subplot(3,2,2),plot(MT,'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(CSLDenoise,'b'),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(b)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('MFF-DenseNet(Without SPP)');axis("auto xy")
subplot(3,2,3),plot(MT,'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(SPPDenoise,'m'),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(c)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('MFF-DenseNet(Without CSL)');axis("auto xy")
subplot(3,2,4),plot(MT,'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(Denoise,'Color',[1.00 0.41 0.16]),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(d)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('MFF-DenseNet');axis("auto xy")
subplot(3,2,5),plot(DenseDenoise,'g')
hold on
plot(CSLDenoise,'b')
hold on
plot(SPPDenoise,'m')
hold on
plot(Denoise,'Color',[1.00 0.41 0.16])
legend('MFF-DenseNet(Without CSL and SPP)','MFF-DenseNet(Without CSL)','MFF-DenseNet(Without SPP)','MFF-DenseNet)'),text('Units','normalized','String','(e)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('Reconstructed signal of different network architecture')
subplot(3,2,6),plot(MT(3001:3100),'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(DenseDenoise(3001:3100),'g')
hold on
plot(CSLDenoise(3001:3100),'b')
hold on
plot(SPPDenoise(3001:3100),'m')
hold on
plot(Denoise(3001:3100),'Color',[1.00 0.41 0.16])
axis('auto xy');
legend('Noise-free segment','MFF-DenseNet(Without CSL and SPP)','MFF-DenseNet(Without CSL)','MFF-DenseNet(Without SPP)','MFF-DenseNet'),text('Units','normalized','String','(f)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
xlabel('Number of sampling points')
title('Enlarged comparisons of reconstructed signal of different network architecture')
