%% ========================================================================
%  Monte-Carlo SER vs OF using B2R2 Reconstruction
%  Noise added in continuous domain + ideal LPF + sampling
% ========================================================================

clear; clc; close all;

%% ================= USER PARAMETERS ======================
num_MC = 100;           % Monte-Carlo trials per OF
OF_list = 2:8;          % oversampling factors
Lambda = 0.4;
Delta  = 2*Lambda;
W = 10;                 % analog bandwidth
E = 10;
SNR_dB = 20;

S = [-1 1 2 -2];        % symbol alphabet
NoS = 10;               % number of symbols (coefficients)

SER_vals = zeros(length(OF_list), num_MC);

fprintf("Running SER Monte Carlo, %d trials per OF...\n", num_MC);


%% ================== MAIN OF LOOP =========================
for oi = 1:length(OF_list)

    OF = OF_list(oi);
    fprintf("\n=========== OF = %d ===========\n", OF);

    % Analog Nyquist quantities
    BW   = 2*pi*W;
    Wnyq = 2*BW;
    Tnyq = 2*pi / Wnyq;      % symbol spacing in time
    Ws   = OF * Wnyq;

    %% ==== Precompute all possible symbol combinations ====
    A = generate_all_as(S, NoS);   % matrix: NoS x |alphabet|^NoS
    num_symbols = size(A,2);

    for mc = 1:num_MC

        %% ---- (1) Choose random symbol vector ----
        idx_true = randi(num_symbols);
        CoS_true = A(:, idx_true);

        %% ---- (2) Generate continuous-time signal ----
        [x_cont, t_d] = generate_analog_transmit(E, W, OF, CoS_true);

        %% ---- (3) Add AWGN ----
        noisy_x = awgn(x_cont, SNR_dB, 'measured');
        x = noisy_x; t = t_d;

        Td = t(2) - t(1);
        Fs = 1/Td;
        N  = length(x);
        f_axis = linspace(-Fs/2, Fs/2, N);

        %% ---- (4) Ideal LPF ----
        X = fftshift(fft(x));
        H = abs(f_axis) <= W;      % ONLY W (your working version)
        H = H(:);
        X_filt = X .* H;
        x_filt = ifft(ifftshift(X_filt));

        %% ---- (5) Sample ----
        [x_samp, t_s] = generate_sampled_signal(x_filt, t_d);
        x_true = x_samp(:);   % discrete true signal
        L = length(x_true);

        %% ---- (6) Modulo fold ----
        x_lambda = mod(x_true + Lambda, 2*Lambda) - Lambda;

        %% ---- (7) True M_lambda (allowed) ----
        delta_true = x_true - x_lambda;
        M = M_lambda(delta_true, Lambda, L);

        %% ---- (8) B2R2 Reconstruction ----
        X_ft_lambda = fft(ifftshift(x_lambda));
        x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L);

        %% ---- (9) Extract symbols from reconstructed x_rec ----
        x_rect = zeros(NoS,1);
        for m = 1:NoS
            t_target = (2*m - NoS)*Tnyq;     % symbol sampling instants
            [~, idx] = min(abs(t_s - t_target));
            x_rect(m) = x_rec(idx);
        end

        %% ---- (10) ML symbol detection ----
        d2 = sum((A - x_rect).^2, 1);   % squared distance to each symbol vector
        [~, idx_est] = min(d2);
        est_symbols = A(:, idx_est);

        %% ---- (11) SER for this trial ----
        SER_vals(oi, mc) = symbol_error_function(CoS_true(:), est_symbols(:));

        if mod(mc,20)==0
            fprintf("   Trial %d/%d   SER = %.3f\n", mc, num_MC, SER_vals(oi,mc));
        end

    end
end


%% ================== FINAL SER STATISTICS ===================
SER_mean = mean(SER_vals, 2);
SER_std  = std(SER_vals, 0, 2);

%% ================== PLOT SER vs OF =========================
figure('Color','w','Position',[200 200 840 420]); hold on; grid on;

errorbar(OF_list, SER_mean, SER_std, '-ob', ...
    'LineWidth',1.8,'MarkerSize',7,'MarkerFaceColor','b');

xlabel("Oversampling Factor (OF)");
ylabel("Symbol Error Rate (SER)");
title("SER vs Oversampling Factor (B2R2, Monte-Carlo)");
set(gca,'FontSize',12);

ylim([0 1]);
xlim([min(OF_list)-0.2, max(OF_list)+0.2]);

legend("Mean SER \pm Std","Location","best");


%% ========================================================================
%                            SUPPORTING FUNCTIONS
% ========================================================================
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
function SER = symbol_error_function(a_true, bestEstimate)
    a_true = a_true(:);
    bestEstimate = bestEstimate(:);
    N = length(a_true);
    SER = sum(a_true ~= bestEstimate) / N;
end

function [x_orig_cont, t_d] = generate_analog_transmit(E, W, OF, CoS)
    NoS = length(CoS);
    BW = 2*pi*W;
    Wnyq = 2*BW;
    Ws = OF * Wnyq;
    Ts = 2*pi / Ws;
    BW   = 2*pi*W;
    Wnyq = 2*BW;
    Tnyq = 2*pi / Wnyq;      % symbol spacing in time
    Ws   = OF * Wnyq;
    Td = Ts / 1000;
    t = (-1:Td:1).';
    f = zeros(length(t),1);
    for m = 1:NoS
        f = f + CoS(m) * sinc( 2*W * (t - ((2*m-(NoS))*Tnyq)));
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

function W = DFT_matrix(N)
    [i,j] = meshgrid(0:N-1,0:N-1);
    omega = exp(-2*pi*1i/N);
    W = omega.^((i - N/2).*j);
end

function delta_q = quant_delta(delta_rec, Lambda)
    delta_q = round(delta_rec/(2*Lambda)) * (2*Lambda);
end

function y = high_pass_matlab(x, beta, rho, w)
    X = fft(ifftshift(x));
    [~, idx_1] = min(abs(w - beta*rho*pi));
    [~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));
    X(1:idx_1) = 0;
    X(idx_2+1:end) = 0;
    y = real(ifftshift(ifft(X)));
end
