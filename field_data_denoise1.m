%% Load data for testing  
Fiednoisydata=load('Field_dataA.dat');
%Change 'Field_dataA.dat' to load different data
%This example was designed to test denoising results using different percentages of the dataset (Figure 21 in the manuscript)
load('MFF_DenseNet10.mat')% 10 percentage
load('MFF_DenseNet20.mat')% 20 percentage
load('MFF_DenseNet30.mat')% 30 percentage
load('MFF_DenseNet.mat')
%% Denoising
MT=Fiednoisydata-mean(Fiednoisydata);
Denoise10=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet10,In);
        DEX=reshape(out,fd,1);
        Denoise10=[Denoise10,a-DEX'];
    end
end

Denoise20=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet20,In);
        DEX=reshape(out,fd,1);
        Denoise20=[Denoise20,a-DEX'];
    end
end

Denoise30=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet30,In);
        DEX=reshape(out,fd,1);
        Denoise30=[Denoise30,a-DEX'];
    end
end

Denoise70=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        Denoise70=[Denoise70,a-DEX'];
    end
end
%% Plot
figure()
subplot(3,2,1),plot(Fiednoisydata(166001:170000),'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(Denoise10(166001:170000),'g'),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(a)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('Ratio=10%')
subplot(3,2,2),plot(Fiednoisydata(166001:170000),'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(Denoise20(166001:170000),'b'),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(b)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('Ratio=20%')
subplot(3,2,3),plot(Fiednoisydata(166001:170000),'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(Denoise30(166001:170000),'m'),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(c)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('Ratio=30%')
subplot(3,2,4),plot(Fiednoisydata(166001:170000),'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(Denoise70(166001:170000),'Color',[1.00 0.41 0.16]),legend('Noisy data','Reconstructed signal'),text('Units','normalized','String','(d)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('Ratio=70%')
subplot(3,2,5),plot(Denoise10(166001:170000),'g')
hold on
plot(Denoise20(166001:170000),'b')
hold on
plot(Denoise30(166001:170000),'m')
hold on
plot(Denoise70(166001:170000),'Color',[1.00 0.41 0.16])
legend('Ratio=10%','Ratio=20%','Ratio=30%','Ratio=70%'),text('Units','normalized','String','(e)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
title('Reconstructed signal using different percentages of training data')
subplot(3,2,6),plot(Fiednoisydata(169101:169300),'Color',[0 0.447058823529412 0.741176470588235])
hold on
plot(Denoise10(169101:169300),'g')
hold on
plot(Denoise20(169101:169300),'b')
hold on
plot(Denoise30(169101:169300),'m')
hold on
plot(Denoise70(169101:169300),'Color',[1.00 0.41 0.16])
legend('Noisy data','Ratio=10%','Ratio=20%','Ratio=30%','Ratio=70%'),text('Units','normalized','String','(f)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
axis([0 200 -7000 5000]),
title('Enlarged comparisons of reconstructed signal using different percentages of training data')
