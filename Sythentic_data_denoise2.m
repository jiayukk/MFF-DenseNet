clear
fd=200;
%% load data and model
load('Noise2.mat');
load('MFF_DenseNet.mat');
MT=Noisydata;DenoiseDense=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(MFF_DenseNet,In);
        DEX=reshape(out,fd,1);
        DenoiseDense=[DenoiseDense,a-DEX'];
    end
end
%% Calculate the spectrum
A=Noise;
% get extracted noise data
D=Noisydata-DenoiseDense;
YA = fft(A);
L=5000;
Fs = 150;
P2 = abs(YA/L);
P1A = P2(1:L/2+1);
f = Fs*(0:(L/2))/L; 
P1A(2:end-1) = 2*P1A(2:end-1);

YD = fft(D);
P2 = abs(YD/L);
P1D = P2(1:L/2+1);
f = Fs*(0:(L/2))/L; 
P1D(2:end-1) = 2*P1D(2:end-1);
%% Figure 10
figure();
subplot(2, 1, 1);
plot(A, 'Color', [0 0 0], 'LineWidth', 1.6, 'DisplayName', 'Original noise data');
hold on;
plot(D, 'Color', [1 0 0], 'LineWidth', 1.4, 'DisplayName', 'Proposed method extracted noise data');
grid on;
xticks(0:1000:5000);yticks(-8000:2000:8000);set(gca, 'GridLineStyle', ':', 'MinorGridLineStyle', '-', 'GridAlpha', 0.2, 'MinorGridAlpha', 0.2, 'Layer', 'top');
xlabel('Number of samepling points')
ylabel('Amplitute/count','FontWeight','bold');
legend('Location', 'southwest', 'NumColumns', 3);
subplot(2, 1, 2);
loglog(f,P1A,'k', 'Color', [0 0 0], 'LineWidth', 1.6, 'DisplayName', 'Original noise data')
hold on;
loglog(f,P1D,'Color', [1 0 0], 'LineWidth', 1.4, 'DisplayName', 'Proposed method extracted noise data')
ylim([0.01, 500]);xlim([0.03,75]);grid on;ylabel('Amplitude(dB)','FontWeight','bold');xlabel('Frequency(Hz)');
%% corc
rho3= corr(A', D', 'Type', 'Spearman');
CORC3 = 1 - (6 * sum((A - D).^2) / (length(D) * (length(D)^2 - 1)));

%r2
mean_observed = mean(A);
TSS = sum((A - mean_observed).^2);
RSS = sum((A - C).^2);
R_squared = 1 - (RSS / TSS);
%CORREHENCE
Fs=150;
[Cxy,f] = mscohere(A,D,[],[],[],Fs);

figure()

plot(f,Cxy,'Color', [1 0 0], 'LineWidth', 1.2)
grid on
xlim([0,75]);ylim([0.5, 1.1]);ylabel('Magnitude-squared coherence','FontWeight','bold');xlabel('Frequency(Hz)');legend('Original noise data-Proposed method')
