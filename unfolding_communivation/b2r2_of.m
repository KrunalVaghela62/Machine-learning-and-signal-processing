%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B2R2 Monte Carlo : MSE vs Oversampling Factor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% ================= PARAMETERS =================

OF_list = [2:8];          % Oversampling factors
MC = 100;               % Monte Carlo runs
W = 10;                 % Analog bandwidth
S = [-1 1 2 -2];        % Symbol set

scale = 100;
E = 10;
Lambda = 0.6;
Delta  = 2*Lambda;
K =fold(@lcm,2:10);
% Fixed dense grid
Td = 1/(2*W*K);

sigma_lpf=Lambda/2;% Modulo parameter
sigma  = sqrt((sigma_lpf^2)/(2*W*Td));

mse_threshold = 1e-10;

mse_avg = zeros(length(OF_list),1);
recovery_rate = zeros(length(OF_list),1);

%% ================= MONTE CARLO LOOP =================

for of_i = 1:length(OF_list)

OF = OF_list(of_i);

fprintf('\nRunning OF = %d\n',OF);

mse_mc = zeros(MC,1);
good_recovery_count = 0;

for mc = 1:MC

%% =========================================================
%% 1) Generate continuous signal
%% =========================================================

[x_orig_cont,t_d,CoS,NoS] = generate_analog_transmit(E,W,OF,S);

%% =========================================================
%% 2) Generate noise
%% =========================================================

Td = t_d(2) - t_d(1);
Fs = 1/Td;

%% =========================================================
%% 3) LOWPASS NOISE
%% =========================================================

noise = sigma * randn(length(t_d),1);

x_noisy_cont = lowpass(noise,W,Fs,ImpulseResponse="iir",Steepness=0.9999);

%% =========================================================
%% 4) Add noise
%% =========================================================

x_orig_cont = x_orig_cont + x_noisy_cont;

idx = (t_d >= -1.8) & (t_d <= 1.8);

t_d = t_d(idx);
x_orig_cont  = x_orig_cont(idx);

%% =========================================================
%% 5) Sample signal
%% =========================================================

[x_sampled,t_s] = generate_sampled_signal(x_orig_cont,t_d,W,OF);

x_true = double(x_sampled(:));
L = length(x_true);

%% =========================================================
%% 6) Modulo folding
%% =========================================================

x_lambda = mod(x_true + Lambda , 2*Lambda) - Lambda;

X_ft_lambda = fft(ifftshift(x_lambda));

%% =========================================================
%% 7) Estimate M
%% =========================================================

delta_true = x_true - x_lambda;

M = M_lambda(delta_true, Lambda, L);

%% =========================================================
%% 8) B2R2 Reconstruction
%% =========================================================

x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L);

%% =========================================================
%% 9) Compute normalized MSE
%% =========================================================

err = x_true - x_rec;

Energy = norm(x_true)^2;

mse_mc(mc) = (norm(err)^2) / Energy;

if mse_mc(mc) < mse_threshold
    good_recovery_count = good_recovery_count + 1;
end
if mod(mc,25)==0
            fprintf("  MC %d / %d\n", mc,MC);
         
             
end
end

%% =========================================================
%% 10) Average MSE
%% =========================================================

mse_avg(of_i) = mean(mse_mc);

recovery_rate(of_i) = good_recovery_count / MC;

fprintf('OF = %d | Good Recoveries = %d / %d\n', ...
        OF, good_recovery_count, MC);

end

%% =========================================================
%% 11) Plot MSE
%% =========================================================

figure;

plot(OF_list,10*log10(mse_avg),'o-','LineWidth',2)

xlabel('Oversampling Factor (OF)')
ylabel('Normalized MSE (dB)')
title('B^2R^2 Reconstruction Performance')

grid on


%% =========================================================
%% 12) Plot Recovery Probability
%% =========================================================

figure

plot(OF_list,recovery_rate,'o-','LineWidth',2)

xlabel('Oversampling Factor (OF)')
ylabel('Recovery Probability')

title('Probability of Perfect Recovery')

grid on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = M_lambda(delta, Lambda, L)

thr = 0.1 * Lambda;

idx = find(abs(delta) > thr);

if isempty(idx)
    M = 0;
    return;
end

m_min = idx(1);
m_max = idx(end);

M = floor(2 * max(m_max - 0.5 * L, 0.5 * L - m_min) + 1);

end


function x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L)

x_lambda = x_lambda(:);

rho = 1 / (OF);

N = floor(M/2);

beta = 1.02;

decay = 0.9999;

reg = 1e-6;

momentum = 0.6;

epsilon = 1e-3 * Lambda;

w = 2*pi*(0:L-1)/L;

[~, idx_1] = min(abs(w - beta*rho*pi));
[~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));

diagonal = ones(L,1);

diagonal(1:idx_1+1) = 0;
diagonal(idx_2:end) = 0;

D = diag(diagonal);

F = DFT_matrix(L);

BB = D * F;

dev_matrix = (BB' * BB);

step_size = 1 / L;

mu = step_size;

mom = momentum;

change = zeros(L,1);

d1 = 0;
d2 = 0;

delta_rec = high_pass_matlab(-x_lambda, beta, rho, w);

center = floor(L/2) + 1;

leftCut = max(1, center - N);
rightCut = min(L, center + N);

delta_rec(1:leftCut-1) = 0;
delta_rec(rightCut+1:end) = 0;

delta_rec = double(delta_rec);

max_iter = 10e4;

for iter = 1:max_iter

vector = x_lambda + delta_rec;

grad = dev_matrix * vector + reg * delta_rec;

change = mom * change + mu * real(grad);

delta_rec = delta_rec - change;

delta_rec(1:leftCut-1) = 0;
delta_rec(rightCut+1:end) = 0;

mu = decay * mu;
mom = decay * mom;

if mod(iter,5) == 0

idxL = leftCut;
idxR = rightCut;

d1_new = delta_rec(idxL);
d2_new = delta_rec(idxR);

if max(abs(d1_new - d1), abs(d2_new - d2)) < epsilon

delta_rec = quant_delta(delta_rec, Lambda);

if N < 1
x_lambda(center) = x_lambda(center) + delta_rec(center);
break;
else

mu = step_size;
mom = momentum;

x_lambda(idxL) = x_lambda(idxL) + delta_rec(idxL);
x_lambda(idxR) = x_lambda(idxR) + delta_rec(idxR);

N = N - 1;

leftCut = max(1, center - N);
rightCut = min(L, center + N);

end

if N >= 0
d1 = delta_rec(leftCut);
d2 = delta_rec(rightCut);
else
d1 = 0;
d2 = 0;
end

else

d1 = d1_new;
d2 = d2_new;

end

end

end

x_rec = x_lambda;

end


function F = DFT_matrix(N)

[k,n] = meshgrid(0:N-1,0:N-1);

F = exp(-2*pi*1i*k.*n/N);

end


function y = high_pass_matlab(x, beta, rho, w)

X_ft_hpf = fft(ifftshift(x));

[~, idx_1] = min(abs(w - beta*rho*pi));
[~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));

X_ft_hpf(1:idx_1+1) = 0;
X_ft_hpf(idx_2:end) = 0;

y = ifftshift(ifft(X_ft_hpf));

y = real(y);

end


function delta_q = quant_delta(delta_rec, Lambda)

delta_q = round(delta_rec/(2*Lambda))*(2*Lambda);

end