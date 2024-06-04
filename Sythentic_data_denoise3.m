clear
%% Load data and model
load("Noise3.mat")
load('MFF_DenseNet.mat')
%% Denoise
fd=200;
MT=ImpulseA;ImpulseC=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        ImpulseC=[ImpulseC,a-DEX'];
    end
end

MT=SquareA;SquareC=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        SquareC=[SquareC,a-DEX'];
    end
end

MT=TriangleA;TriangleC=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        TriangleC=[TriangleC,a-DEX'];
    end
end

MT=MixedA;MixedC=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        MixedC=[MixedC,a-DEX'];
    end
end

%% Figure 7
figure()
subplot 341,plot(TriangleA(1:5000),'Color',[0 0.447058823529412 0.741176470588235]),hold on
plot(TriangleD(1:5000),'k'),hold on
plot(TriangleC(1:5000),'Color',[1.00 0.41 0.16],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Triangle noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(a)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 345,plot(TriangleA(5001:10000),'Color',[0 0.447058823529412 0.741176470588235]),hold on
plot(TriangleD(5001:10000),'k'),hold on
plot(TriangleC(5001:10000),'Color',[1.00 0.41 0.16],'LineStyle','--')
axis([0 5000 -7000 5000]),legend('Triangle noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(e)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 349,plot(TriangleA(10001:15000),'Color',[0 0.447058823529412 0.741176470588235]),hold on
plot(TriangleD(10001:15000),'k'),hold on
plot(TriangleC(10001:15000),'Color',[1.00 0.41 0.16],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Triangle noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(i)','Position',[0.01 0.93 0]);
ylabel('Amplitute/count','FontWeight','bold');xlabel('Number og sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 342,plot(ImpulseA(1:5000),'Color',[0.47,0.67,0.19]),hold on
plot(ImpulseD(1:5000),'k'),hold on
plot(ImpulseC(1:5000),'Color',[1.00 0.41 0.16],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Impulse noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(b)','Position',[0.01 0.93 0]);
grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 346,plot(ImpulseA(10001:15000),'Color',[0.47,0.67,0.19]),hold on
plot(ImpulseD(10001:15000),'k'),hold on
plot(ImpulseC(10001:15000),'Color',[1.00 0.41 0.16],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Impulse noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(f)','Position',[0.01 0.93 0]);
grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot (3,4,10),plot(ImpulseA(15001:20000),'Color',[0.47,0.67,0.19]),hold on
plot(ImpulseD(15001:20000),'k'),hold on
plot(ImpulseC(15001:20000),'Color',[1.00 0.41 0.16],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Impulse noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(j)','Position',[0.01 0.93 0]);
xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 343,plot(SquareA(1:5000),'Color',[0.72,0.27,1.00]),hold on
plot(SquareD(1:5000),'k'),hold on
plot(SquareC(1:5000),'Color',[0.85,0.33,0.10],'LineStyle','--')
axis([0 5000 -6000 6000]),legend('Square noisy data','Noise-free signal','Dnoised signal'),text('Units','normalized','String','(c)','Position',[0.01 0.93 0]);
grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 347,plot(SquareA(10001:15000),'Color',[0.72,0.27,1.00]),hold on
plot(SquareD(10001:15000),'k'),hold on
plot(SquareC(10001:15000),'Color',[0.85,0.33,0.10],'LineStyle','--')
axis([0 5000 -10000 6000]),legend('Square noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(g)','Position',[0.01 0.93 0]);
grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot (3,4,11),plot(SquareA(15001:20000),'Color',[0.72,0.27,1.00]),hold on
plot(SquareD(15001:20000),'k'),hold on
plot(SquareC(15001:20000),'Color',[0.85,0.33,0.10],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Square noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(k)','Position',[0.01 0.93 0]);
xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 344,plot(MixedA(1:5000),'Color',[0.00,0.00,1.00]),hold on
plot(MixedD(1:5000),'k'),hold on
plot(MixedC(1:5000),'Color',[0.85,0.33,0.10],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Mixed noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(d)','Position',[0.01 0.93 0]);
grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 348,plot(MixedA(10001:15000),'Color',[0.00,0.00,1.00]),hold on
plot(MixedD(10001:15000),'k'),hold on
plot(MixedC(10001:15000),'Color',[0.85,0.33,0.10],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Mixed noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(h)','Position',[0.01 0.93 0]);
grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot (3,4,12),plot(MixedA(15001:20000),'Color',[0.00,0.00,1.00]),hold on
plot(MixedD(15001:20000),'k'),hold on
plot(MixedC(15001:20000),'Color',[0.85,0.33,0.10],'LineStyle','--')
axis([0 5000 -5000 5000]),legend('Mixed noisy data','Noise-free signal','Denoised signal'),text('Units','normalized','String','(l)','Position',[0.01 0.93 0]);
xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');

