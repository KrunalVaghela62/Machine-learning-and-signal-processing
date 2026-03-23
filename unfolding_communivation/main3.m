% run_b2r2_from_continuous.m
% Uses your continuous generator -> downsample -> B2R2 recovery (MATLAB)
clear; close all; clc;

%% ===================== User parameters =====================
OF = 5;                 % oversampling factor (your continuous generator uses this)
%coeff_num = 10;         % number of sinc components in continuous generator
Lambda = 0.2; % modulo parameter

scale = 100;            % plotting window half-width
E = 10;                 % passed to generate_original_signal (kept for continuity)
W = 10; 
K =fold(@lcm,2:10);
% Fixed dense grid
Td = 1/(2*W*K);

sigma_lpf=Lambda/8;
sigma  = sqrt((sigma_lpf^2)/(2*W*Td));% analog W (not used inside B2R2 directly, but used by generator)

S=[-1,1,2,-2];
BW = 2 * pi * W;         % Bandwidth (rad/s)
c=1;
delta=2*Lambda;
% Derived parameters
Wnyq = 2 * BW;
Tnyq = 2 * pi / Wnyq;    
Ws   = OF * Wnyq;
Ts   = 2 * pi / Ws;
%Td   = Ts / 1000;
%% ================ 1) Generate continuous signal & sample ============
[x_orig_cont,t_d,CoS,NoS] = generate_analog_transmit(E, W, OF, S);
CoS
A = generate_all_as(S, NoS);
%x_lambda = mod(x_orig_cont + Lambda, 2*Lambda) - Lambda;    % folded samples
%% ================= Generate continuous signal =================
figure;
plot(t_d,x_orig_cont,'LineWidth',1.5)
hold on

title('no noise1')
xlabel('T')
ylabel('|X(T)|')
grid on

% Nyquist sampling locations
t_stem = (2*(1:NoS) - NoS) * Tnyq;

% Signal values at those locations
x_stem = CoS;

stem(t_stem, x_stem,'r','filled','LineWidth',1.5)
%h = 2*W * sinc(2*W*t_d);   % ideal impulse response
noise = sigma * randn(length(t_d),1);
Td=t_d(2)-t_d(1);
Fs=1/Td;
x_noisy_cont=lowpass(noise,W,Fs,ImpulseResponse="iir",Steepness=0.9999);
%% 3) Sample
%{
x_noisy_cont = conv(noise, h, 'same') * Td;
%}
x_orig_cont=x_orig_cont+x_noisy_cont;
idx = (t_d >= -0.8) & (t_d <= 0.8);

t_d = t_d(idx);
x_orig_cont = x_orig_cont(idx);
x_noisy_cont = x_noisy_cont(idx);
figure;
plot(t_d,x_orig_cont,'LineWidth',1.5)
title('no noise')
xlabel('T')
ylabel('|X(T)|')
grid on
figure;
plot(t_d,x_noisy_cont,'LineWidth',1.5)
title(' noise')
xlabel('T')
ylabel('|X(T)|')
grid on
%% ================= Modulo operation =================
%x_lambda1 = mod(x_orig_cont + 0.205, 2*0.205) - 0.205;
%% ================= Sampling parameters =================
%x_orig_cont=awgn(x_orig_cont,20,'measured');
%{
noisy_x=awgn(x_orig_cont,20,'measured');
x = noisy_x;          % noisy signal
t = t_d;

Td = t(2) - t(1);
Fs = 1/Td;
N  = length(x);

X = fftshift(fft(x));    % frequency-domain signal
plot_spectrum(noisy_x, t_d);
figure;
plot(t, real(x));
title('Filtered Signal after adding noise');
grid on;
% --- Ideal low-pass at cutoff W Hz ---
x = noisy_x;          % noisy signal
t = t_d;

Td = t(2) - t(1);
Fs = 1/Td;
N  = length(x);

X = fftshift(fft(x));    % frequency-domain signal

% Frequency axis
f_axis = linspace(-Fs/2, Fs/2, N);

% Ideal low-pass mask
cutoff = W;     % 10 Hz
H = abs(f_axis) <= Ws/2;   % mask 1 inside band, 0 outside
H = H(:);        % force column vector
% Apply LPF
X_filt = X .* H;

% Back to time domain
x_filt = ifft(ifftshift(X_filt));



%}
Lambda1=0.25;

[x_sampled, t_s] = generate_sampled_signal(x_orig_cont, t_d,W,OF); % downsample by 1000 as before

% Use the sampled vector as the discrete-time input
x_true = double(x_sampled(:));    % make column vector
L = length(x_true);
n = (-floor(L/2) : ceil(L/2)-1).'; % index axis, used for plotting

fprintf('Using sampled discrete length L = %d\n', L);

%% ================= 2) Modulo (fold) operation =======================
x_lambda = mod(x_true + Lambda, 2*Lambda) - Lambda;    % folded samples
X_ft_lambda = fft(ifftshift(x_lambda));

%% ================ 3) Estimate delta and M (folded count) ============
delta_true = x_true - x_lambda;     % true residual (simulation only)
M = M_lambda(delta_true, Lambda, L);
fprintf('Estimated number of folded samples M = %d\n', M);
%% ================ 4) Run B2R2 (MATLAB BBRR) =========================
% Important: when using your sampled analog signal, the normalized discrete-time
% bandwidth for B2R2 is rho = 1/(2*OF) (derived from your sampling/downsampling).
x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L);
x_rect = zeros(NoS,1);
for m = 1:NoS
    t_target = (2*m - NoS)*Tnyq;
    [~, idx2] = min(abs(t_s - t_target));
    x_rect(m) = x_rec(idx2);
end

dB = sum((A - x_rect).^2,1);
[~, idxB] = min(dB);

estB = A(:, idxB)
%% ================ 5) Evaluate & Plot ==================================
err = x_true - x_rec;
% Energy normalization similar to Python (E_x_n)
Energy = norm(x_true)^2;
mse = (norm(err)^2) / Energy;


fprintf('MSE = %.6e, MSE(dB) = %.2f dB\n', mse, 10*log10(mse));

center = floor(L/2)+1;
if center-scale < 1; scale = center-1; end
if center+scale > L; scale = min(scale, L-center); end



figure;
plot(n(center-scale:center+scale), x_lambda(center-scale:center+scale), '-k','LineWidth',1.1); hold on;
plot(n(center-scale:center+scale), x_true(center-scale:center+scale), '--r','LineWidth',1.1);
legend('Modulo','Original');
title('Modulo vs Original (center window)'); grid on;
figure;
plot(t_s(center-scale:center+scale), x_true(center-scale:center+scale), '-k','LineWidth',1.1);
hold on;
plot(t_s(center-scale:center+scale), x_rec(center-scale:center+scale), '--r','LineWidth',1.1);

xlabel('Time')
ylabel('Signal')
legend('true','recovered')
title('true vs recovered (time axis)')
grid on



function M = M_lambda(delta, Lambda, L)
    % same logic as Python: threshold 0.1*Lambda and compute spread
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
    % MATLAB translation of the Python reconstruction_method.BBRR
    % This version uses rho = 1/(2*OF) to match your sampling.
    x_lambda = x_lambda(:);
    rho = 1 / (OF);    % <<< IMPORTANT: note factor 2 here for your sampled analog signal
    N = floor(M/2);
    beta = 1.02;
    decay = 0.9999;
    reg = 1e-6;
    momentum = 0.6;
    epsilon = 1e-3 * (Lambda^(1.5) * OF);
    w = linspace(0, 2*pi, L);   % normalized frequency vector like Python

    % Build the frequency mask (diagonal) for high-pass band
    [~, idx_1] = min(abs(w - beta*rho*pi));
    [~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));
    diagonal = ones(L,1);
    diagonal(1:idx_1+1) = 0;
    diagonal(idx_2:end) = 0;
    D = diag(diagonal);
    F = DFT_matrix(L);   % potentially large; ok for L ~ 1024
    BB = D * F;
    dev_matrix = (BB' * BB);

    if OF == 2
        step_size = 2 / L;
    else
        step_size = 2 / (idx_2 - idx_1);
    end

    % initialization
    mu = step_size;
    mom = momentum;
    change = zeros(L,1);
    d1 = 0; d2 = 0;

    % high_pass initialization (MATLAB implementation)
    delta_rec = high_pass_matlab(-x_lambda, beta, rho, w);
    center = floor(L/2) + 1;
    % projection within initial support (center +/- N)
    leftCut = max(1, center - N);
    rightCut = min(L, center + N);
    delta_rec(1 : leftCut-1) = 0;
    delta_rec(rightCut+1 : end) = 0;
    delta_rec = double(delta_rec);

    max_iter = 10e6;
    for iter = 1:max_iter
        % PGD step
        vector = x_lambda + delta_rec;
        grad = dev_matrix * vector + reg * delta_rec;
        change = mom * change + mu * real(grad);
        delta_rec = delta_rec - change;

        % Projection: zero outside support (center +- N)
        delta_rec(1 : leftCut-1) = 0;
        delta_rec(rightCut+1 : end) = 0;

        % decay params
        mu = decay * mu;
        mom = decay * mom;

        if mod(iter,5) == 0
            % Compare boundary changes
            idxL = leftCut;
            idxR = rightCut;
            d1_new = delta_rec(idxL);
            d2_new = delta_rec(idxR);
            if max(abs(d1_new - d1), abs(d2_new - d2)) < epsilon
                % Quantize delta
                delta_rec = quant_delta(delta_rec, Lambda, leftCut, rightCut);
                %[~, i] = min(abs(c_n - delta_rec(idxL)));
                %delta_rec(idxL) = c_n(i);
                %[~, i] = min(abs(c_n - delta_rec(idxR)));
                %delta_rec(idxR) = c_n(i);
                if N < 1
                    x_lambda(center) = x_lambda(center) + delta_rec(center);
                    break;
                else
                    mu = step_size;
                    mom = momentum;
                    x_lambda(idxL) = x_lambda(idxL) + delta_rec(idxL);
                    x_lambda(idxR) = x_lambda(idxR) + delta_rec(idxR);
                    % shrink support
                    N = N - 1;
                    leftCut = max(1, center - N);
                    rightCut = min(L, center + N);
                end
                % update d1,d2 for next stage
                if N >= 0
                    d1 = delta_rec(leftCut);
                    d2 = delta_rec(rightCut);
                else
                    d1 = 0; d2 = 0;
                end
            else
                d1 = d1_new; d2 = d2_new;
            end
        end
    end

    % final correction
x_rec = x_lambda;
end

function W = DFT_matrix(N)
    % Construct DFT-like matrix used by Python code (matching indexing)
    [i,j] = meshgrid(0:N-1, 0:N-1);
    omega = exp(-2*pi*1i / N);
    W = omega .^ ((i - N/2) .* j);   % matches python formula
end

function y = high_pass_matlab(x, beta, rho, w)
    % MATLAB version of high_pass (matching python)
    % x is length-L column vector
    X_ft_hpf = fft(ifftshift(x));
    [~, idx_1] = min(abs(w - beta * rho * pi));
    [~, idx_2] = min(abs(w - (2 * pi - beta * rho * pi)));
    X_ft_hpf(1:idx_1) = 0;
    X_ft_hpf(idx_2+1:end) = 0;
    y = ifftshift(ifft(X_ft_hpf));
    y = real(y);
end

function delta_rec = quant_delta(delta_rec, Lambda, leftCut, rightCut)

for i = [leftCut rightCut]


k = round(delta_rec(i) / (2*Lambda));
delta_rec(i) = k * (2*Lambda);


end

end

function [c_n,n]=possible_z_values(c,delta)%z=delta*c_n
    n = floor((c/delta)+0.5);
    c_n=-n:n;
end

