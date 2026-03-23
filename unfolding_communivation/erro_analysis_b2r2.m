%% ========================================================================
%  Monte-Carlo B2R2 Performance vs Oversampling Factor (OF)
%  Noise added in continuous domain + ideal LPF + sampling
% ========================================================================

clear; clc; close all;

%% ================= USER PARAMETERS ======================
num_MC = 100;           % Monte-Carlo trials per OF
OF_list = 2:8;         % oversampling factors
Lambda = 0.4;           % modulo parameter
Delta = 2*Lambda;
W = 10;                 % analog bandwidth (Hz)
E = 10;                 % amplitude parameter
SNR_dB = 20;
S = [-1 1 2 -2];        % generator symbol choices
NoS = 20;               % number of sinc coefficients

MC_MSE_dB = zeros(length(OF_list), num_MC);

fprintf("Running Monte Carlo, %d trials per OF...\n", num_MC);

%% ================= MAIN LOOP =============================
for oi = 1:length(OF_list)
    
    OF = OF_list(oi);
    fprintf("\n==========================\n");
    fprintf("Processing OF = %d\n", OF);
    fprintf("==========================\n");
    
    %% Derived quantities
    BW = 2*pi*W;
    Wnyq = 2*BW;
    Ws = OF * Wnyq;

    for mc = 1:num_MC
        
        %% ---- 1) Generate continuous-time bandlimited signal ----
        CoS = S(randi(numel(S), NoS, 1));    % true coefficients
        [x_cont, t_d] = generate_analog_transmit(E, W, OF, CoS);

        %% ---- 2) Add AWGN ----
        noisy_x = awgn(x_cont, SNR_dB, 'measured');
        x = noisy_x;
        t = t_d;

        Td = t(2) - t(1);
        Fs = 1 / Td;
        N  = length(x);
        f_axis = linspace(-Fs/2, Fs/2, N);

        %% ---- 3) Ideal LPF in frequency domain (same as your working script) ----
        X = fftshift(fft(x));
        H = abs(f_axis) <= W;   % LPF mask
        H = H(:);
        X_filt = X .* H;
        x_filt = ifft(ifftshift(X_filt));

        %% ---- 4) Sample filtered continuous-time signal ----
        [x_sampled, t_s] = generate_sampled_signal(x_filt, t_d);
        x_true = x_sampled(:);
        L = length(x_true);

        %% ---- 5) Modulo folding ----
        x_lambda = mod(x_true + Lambda, 2*Lambda) - Lambda;

        %% ---- 6) M_lambda detection (cheating allowed) ----
        delta_true = x_true - x_lambda;
        M = M_lambda(delta_true, Lambda, L);

        %% ---- 7) B2R2 Reconstruction ----
        X_ft_lambda = fft(ifftshift(x_lambda));
        x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L);

        %% ---- 8) Compute MSE ----
        err = x_true - x_rec;
        mse = sum(err.^2) / sum(x_true.^2);
        MC_MSE_dB(oi, mc) = 10*log10(mse);

        if mod(mc,20)==0
            fprintf("   Trial %d/%d  MSE(dB)=%.2f\n", mc, num_MC, MC_MSE_dB(oi,mc));
        end
    end
end

%% ================= POST-PROCESSING =======================
mean_MSE = mean(MC_MSE_dB, 2);
std_MSE  = std(MC_MSE_dB,  0, 2);

%% ================= PLOT RESULTS ===========================
figure('Color','w','Position',[200 200 840 420]); hold on; grid on;

upper = mean_MSE + std_MSE;
lower = mean_MSE - std_MSE;

fill([OF_list fliplr(OF_list)], ...
     [upper' fliplr(lower')], ...
     [0.7 0.8 1], 'EdgeColor','none','FaceAlpha',0.3);

plot(OF_list, mean_MSE, '-ob', 'LineWidth',2, ...
     'MarkerFaceColor','b','MarkerSize',7);

xlabel("Oversampling Factor (OF)");
ylabel("MSE (dB)");
title("B2R2 Reconstruction Performance vs Oversampling Factor");
set(gca,'FontSize',12);
xlim([min(OF_list)-0.2, max(OF_list)+0.2]);
legend("Mean ± Std","Mean MSE","Location","best");

%% ========================================================================
%                            SUPPORTING FUNCTIONS
% ========================================================================

function [x_orig_cont, t_d] = generate_analog_transmit(E, W, OF, CoS)
    NoS = length(CoS);
    BW = 2*pi*W;
    Wnyq = 2*BW;
    Ws = OF * Wnyq;
    Ts = 2*pi / Ws;
    Td = Ts / 1000;
    t = (-1:Td:1).';
    f = zeros(length(t),1);
    for m = 1:NoS
        f = f + CoS(m) * sinc(2*W*t - Ts*(m)*10);
    end
    x_orig_cont = f;
    t_d = t;
end

function [x_samp, t_s] = generate_sampled_signal(x_cont, t_d)
    x_samp = downsample(x_cont, 1000);
    t_s = downsample(t_d,   1000);
end

function M = M_lambda(delta, Lambda, L)
    thr = 0.1 * Lambda;
    idx = find(abs(delta) > thr);
    if isempty(idx)
        M = 0; return;
    end
    m_min = idx(1);
    m_max = idx(end);
    M = floor(2 * max(m_max - 0.5*L, 0.5*L - m_min) + 1);
end

function x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L)

    x_lambda = x_lambda(:);
    rho = 1 / OF;
    N = floor(M/2);
    beta = 1.05;
    decay = 0.999;
    reg = 1e-6;
    momentum = 0.9;
    epsilon = 1e-3 * (Lambda^1.5 * OF);
    w = linspace(0,2*pi,L);

    [~, idx_1] = min(abs(w - beta*rho*pi));
    [~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));

    diagMask = ones(L,1);
    diagMask(1:idx_1+1) = 0;
    diagMask(idx_2:end) = 0;

    D = diag(diagMask);
    F = DFT_matrix(L);
    dev_matrix = (D*F)'*(D*F);

    if OF==2
        step_size = 2/L;
    else
        step_size = 2/(idx_2 - idx_1);
    end

    mu = step_size;
    mom = momentum;
    change = zeros(L,1);

    center = floor(L/2)+1;
    delta_rec = high_pass_matlab(-x_lambda, beta, rho, w);
    leftCut = max(1, center-N);
    rightCut = min(L, center+N);
    delta_rec(1:leftCut-1) = 0;
    delta_rec(rightCut+1:end) = 0;

    d1=0; d2=0;

    for iter = 1:5e5
        vector = x_lambda + delta_rec;
        grad = dev_matrix*vector + reg*delta_rec;
        change = mom*change + mu*real(grad);
        delta_rec = delta_rec - change;

        delta_rec(1:leftCut-1) = 0;
        delta_rec(rightCut+1:end) = 0;

        mu = decay*mu;
        mom = decay*mom;

        if mod(iter,5)==0
            d1_new = delta_rec(leftCut);
            d2_new = delta_rec(rightCut);

            if max(abs(d1_new-d1), abs(d2_new-d2)) < epsilon
                delta_rec = quant_delta(delta_rec, Lambda);

                if N < 1
                    x_lambda(center) = x_lambda(center) + delta_rec(center);
                    break;
                else
                    mu = step_size; mom = momentum;
                    x_lambda(leftCut)  = x_lambda(leftCut)  + delta_rec(leftCut);
                    x_lambda(rightCut) = x_lambda(rightCut) + delta_rec(rightCut);
                    N = N-1;
                    leftCut = max(1, center-N);
                    rightCut = min(L, center+N);
                end

                if N>=0
                    d1 = delta_rec(leftCut);
                    d2 = delta_rec(rightCut);
                else
                    d1=0; d2=0;
                end

            else
                d1 = d1_new; d2 = d2_new;
            end
        end
    end

    x_rec = x_lambda;
end

function W = DFT_matrix(N)
    [i,j] = meshgrid(0:N-1,0:N-1);
    omega = exp(-2*pi*1i/N);
    W = omega.^((i - N/2).*j);
end

function y = high_pass_matlab(x, beta, rho, w)
    X = fft(ifftshift(x));
    [~, idx_1] = min(abs(w - beta*rho*pi));
    [~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));
    X(1:idx_1) = 0;
    X(idx_2+1:end) = 0;
    y = real(ifftshift(ifft(X)));
end

function delta_q = quant_delta(delta_rec, Lambda)
    delta_q = round(delta_rec/(2*Lambda)) * (2*Lambda);
end
