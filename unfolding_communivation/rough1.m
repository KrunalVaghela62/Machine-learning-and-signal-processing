clear; close all; clc;

%% PARAMETERS
W  = 10;            
Fs = 2000;          
T  = 2;             
N  = Fs*T;
Lambda=0.2;
t  = linspace(-T/2, T/2, N);
Ts = 1/Fs;
K =fold(@lcm,2:10);
% Fixed dense grid
Td = 1/(2*W*K);
Fs=1/Td
T  = 2;  
N  = Fs*T;

t  = linspace(-T/2, T/2, N);

sigma_lpf=Lambda/4;% Modulo parameter
sigma  = sqrt((sigma_lpf^2)/(2*W*Td));

%% Input: x(t) = W rect(Wt)
%x = W * (abs(2*W*t) <= 1);
p=10e8;
x = p*sinc(p*t);
%% Ideal LPF impulse response (cutoff = 2W Hz)
h = 2*W * sinc(2*W*t)*Ts;   % MATLAB sinc = sin(pi x)/(pi x)

%% Convolution (approx continuous convolution)
%y = conv(x, h, 'same');
%y=lowpass(x,W,Fs,ImpulseResponse="iir",Steepness=0.9999);
%% Ideal LPF using sinc convolution

Td = t(2)-t(1);   % time step

h = 2*W * sinc(2*W*t);   % ideal impulse response

y = conv(x, h, 'same') * Td;
%% FFTs
X = fftshift(fft(x))*Td;
Y = fftshift(fft(y))*Td;
f = linspace(-Fs/2, Fs/2, N);

%% Plot comparison
figure;

subplot(2,1,1)
plot(f, abs(X), 'LineWidth',1.5)
title('Original Spectrum')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-500 500])
grid on

subplot(2,1,2)
plot(f, abs(Y), 'LineWidth',1.5)
title('After Ideal Sinc LPF (cutoff = 2W)')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
xlim([-500 500])
grid on