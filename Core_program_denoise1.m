clear
%% Load data and model
load('figure9.mat')
load('MFF_DenseNet.mat')

%% Denoising 
fd=200;
MT=Noise_sample9+Signal_sample9;Denoisedata=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        Denoisedata=[Denoisedata,a-DEX'];
    end
end
C=Denoisedata;
%% Cauculating spectra
fs=150;
Y1 = fft(A);
L=5000;
P1 = abs(Y1/L);
P1A = P1(1:L/2+1);
f = fs*(0:(L/2))/L; 
P1A(2:end-1) = 2*P1A(2:end-1);

Y2 = fft(B);
P2 = abs(Y2/L);
P2A = P2(1:L/2+1);
P2A(2:end-1) = 2*P2A(2:end-1);

Y3 = fft(C);
P3 = abs(Y3/L);
P3A = P3(1:L/2+1);
P3A(2:end-1) = 2*P3A(2:end-1);


Y4 = fft(D);
P4 = abs(Y4/L);
P4A = P4(1:L/2+1);
P4A(2:end-1) = 2*P4A(2:end-1);

%% CWT
sig1 =A;                       
t = (1:length(sig1))/fs;
fb = cwtfilterbank(SignalLength=length(sig1), ...
    SamplingFrequency=fs, ...
    VoicesPerOctave=12);
[cfs1,frq1] = wt(fb,sig1);

sig2 =B;
fb = cwtfilterbank(SignalLength=length(sig2), ...
    SamplingFrequency=fs, ...
    VoicesPerOctave=12);
[cfs2,frq2] = wt(fb,sig2);

sig3 =C;                        
fb = cwtfilterbank(SignalLength=length(sig3), ...
    SamplingFrequency=fs, ...
    VoicesPerOctave=12);
[cfs3,frq3] = wt(fb,sig3);

sig4 =D;                       
fb = cwtfilterbank(SignalLength=length(sig4), ...
    SamplingFrequency=fs, ...
    VoicesPerOctave=12);
[cfs4,frq4] = wt(fb,sig4);

cfsa=[cfs1,cfs2,cfs3,cfs4];
minvalue=min(cfsa(:));
maxvalue=max(cfsa(:));
cfss1=(cfs1 - minvalue) / (maxvalue - minvalue);
cfss2=(cfs2 - minvalue) / (maxvalue - minvalue);
cfss3=(cfs3 - minvalue) / (maxvalue - minvalue);
cfss4=(cfs4 - minvalue) / (maxvalue - minvalue);

%% Plot
figure()
subplot 341,plot(A,'k'),axis([0 5000 -10000 10000]),text('Units','normalized','String','(a)','Position',[0.01 0.93 0]); title('Noisy data');
ylabel('Amplitude/count','FontWeight','bold');xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 345,loglog(f,P1A,'k'),xlim([0.03,75]);title('Spectrum of (a)'); ylim([0.001, 10000]);grid on; text('Units','normalized','String','(e)','Position',[0.01 0.93 0]);ylabel('Amplitude(dB)','FontWeight','bold');xlabel('Frequency(Hz)');
subplot 349,pcolor(t,frq1,abs(cfss1));colormap("jet"); title("Scalogram of (a)");set(gca,"yscale","log");xlabel('Time (s)','FontWeight','bold');
text('Units', 'normalized', 'String', '(i)', 'Position', [0.01 1.08 0]);hcol = colorbar;set(hcol);ylabel(hcol, 'Magnitude','FontWeight','bold');
shading interp  ;axis tight;ylabel("Frequency (Hz)",'FontWeight','bold');set(gca, 'FontSize', 12);clim([0,1])
subplot 342,plot(B,'Color',[0 0.447058823529412 0.741176470588235]),axis([0 5000 -8000 8000]),text('Units','normalized','String','(b)','Position',[0.01 0.93 0]);title('Extracted noise');
ylabel('Amplitude/count','FontWeight','bold');xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 346,loglog(f,P2A,'Color',[0 0.447058823529412 0.741176470588235]),xlim([0.03,75]);title('Spectrum of (b)'); ylim([0.001, 1000]);grid on; text('Units','normalized','String','(f)','Position',[0.01 0.93 0]);ylabel('Amplitude(dB)','FontWeight','bold');xlabel('Frequency(Hz)');
subplot(3,4,10);pcolor(t,frq2,abs(cfss2));colormap("jet"); title("Scalogram of (b)");set(gca,"yscale","log");xlabel('Time (s)','FontWeight','bold');
text('Units', 'normalized', 'String', '(j)', 'Position', [0.01 1.08 0]);hcol = colorbar;set(hcol)%;ylabel(hcol, 'Magnitude','FontWeight','bold');
shading interp  ;axis tight;set(gca, 'FontSize', 12);clim([0,1])
subplot 344,plot(C,'Color',[0.85,0.33,0.10]),axis([0 5000 -1000 1000]),text('Units','normalized','String','(d)','Position',[0.01 0.93 0]);title('Denoised signal');
ylabel('Amplitude/count','FontWeight','bold');xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 348,loglog(f,P3A,'Color',[0.85,0.33,0.10]),ylim([0.001, 10000]);xlim([0.03,75]);grid on;title('Spectrum of (d)'); text('Units','normalized','String','(h)','Position',[0.01 0.93 0]);ylabel('Amplitude(dB)','FontWeight','bold');xlabel('Frequency(Hz)');
subplot(3,4,12);pcolor(t,frq3,abs(cfss3));colormap("jet"); title("Scalogram of (d)");set(gca,"yscale","log");xlabel('Time (s)','FontWeight','bold');
text('Units', 'normalized', 'String', '(l)', 'Position', [0.01 1.08 0]);hcol = colorbar;set(hcol);ylabel(hcol, 'Magnitude','FontWeight','bold');
shading interp  ;axis tight;set(gca, 'FontSize', 12);clim([0,0.4])
subplot 343,plot(D,'Color',[0.47,0.67,0.19]),axis([0 5000 -1000 1000]),text('Units','normalized','String','(c)','Position',[0.01 0.93 0]);title('Oringinal signal');
ylabel('Amplitude/count','FontWeight','bold');xlabel('Number of sampling points');grid on;set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
subplot 347,loglog(f,P4A,'Color',[0.47,0.67,0.19]),ylim([0.001, 10000]);title('Spectrum of (c)'); xlim([0.03,75]);grid on; text('Units','normalized','String','(g)','Position',[0.01 0.93 0]);ylabel('Amplitude(dB)','FontWeight','bold');xlabel('Frequency(Hz)');
subplot(3,4,11);pcolor(t,frq4,abs(cfss4));colormap("jet"); title("Scalogram of (c)");set(gca,"yscale","log");xlabel('Time (s)','FontWeight','bold');
text('Units', 'normalized', 'String', '(k)', 'Position', [0.01 1.08 0]);hcol = colorbar;set(hcol);ylabel(hcol, 'Magnitude','FontWeight','bold');
shading interp  ;axis tight;set(gca, 'FontSize', 12);clim([0,0.4])
