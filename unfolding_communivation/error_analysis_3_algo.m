%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Monte-Carlo SER vs OF
% Compare:
%   (A) Exhaustive ML (on folded samples)
%   (B) B2R2 + ML
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
%% ================= USER PARAMETERS =========================
OF_list = [21];
num_MC  = 70;

S  = [-1 1 2 -2];
E  = 10;
W  = 10;
K = 5040;
Td = 1/(2*W*K);
lambda = 0.8;
Delta  = 2*lambda;
sigma_lpf=0.5/1.7;% Modulo parameter
sigma  = sqrt((sigma_lpf^2)/(2*W*Td));

SER_ML   = zeros(length(OF_list), num_MC);
SER_B2R2 = zeros(length(OF_list), num_MC);
SER_no_fold   = zeros(length(OF_list), num_MC);

good_recovery = zeros(length(OF_list),1);   % number of perfect B2R2 reconstructions
mse_threshold = 1e-10;
figure; hold on; grid on;
xlabel("Oversampling Factor (OF)");
ylabel("Symbol Error Rate (SER)");
ylim([0 100]);

%% ================= MAIN LOOP ================================
for of_idx = 1:length(OF_list)

    OF = OF_list(of_idx);
    fprintf("\n========== OF = %d ==========\n", OF);

    %% --- Build codebook ONCE per OF ---
    [x_tmp, t_d, ~, NoS] = generate_analog_transmit(E, W, OF, S);
    idx = (t_d >= -0.8) & (t_d <= 0.8);
    t_d_crop = t_d(idx);

    [~, t_s] = generate_sampled_signal(x_tmp, t_d_crop, W, OF);

    A = generate_all_as(S, NoS);
    H = build_H_matrix(W, NoS, t_s);
    x_all = H*A;
    
    %% ================= MONTE CARLO ==========================
    for mc = 1:num_MC

        %% 1) Generate clean signal
        [x_cont, t_d, CoS_true, ~] = ...
            generate_analog_transmit(E, W, OF, S);

        Td = t_d(2)-t_d(1);
        Fs = 1/Td;
        
        %% 2) Add noise + LPF
        noise = sigma*randn(length(t_d),1);
        x_noise = lowpass(noise,W,Fs,ImpulseResponse="iir",Steepness=0.9999);
        
        
        x_cont = x_cont + x_noise;


        %% 3) Crop & sample
        idx = (t_d >= -0.8) & (t_d <= 0.8);
        t_d = t_d(idx);
        x_cont = x_cont(idx);

        
        [x_samp, t_s] = ...
            generate_sampled_signal(x_cont, t_d, W, OF);
        x_samp = x_samp(:);
        
        %% =====================================================
        % A) Exhaustive ML
        %% =====================================================
        %x_all = mod(x_all + lambda, 2*lambda) - lambda;
        dA = sum((x_all - x_samp).^2,1);
        [~, idxC] = min(dA);

        estC = A(:, idxC);
        SER_no_fold(of_idx, mc) = ...
            symbol_error_function(CoS_true(:), estC(:));

        %% 4) Modulo
        x_mod = mod(x_samp + lambda, 2*lambda) - lambda;

        %% =====================================================
        % A) Exhaustive ML
        %% =====================================================
        x_all_mod = mod(x_all + lambda, 2*lambda) - lambda;
        dA = sum((x_all_mod - x_mod).^2,1);
        [~, idxA] = min(dA);

        estA = A(:, idxA);
        SER_ML(of_idx, mc) = ...
            symbol_error_function(CoS_true(:), estA(:));

        %% =====================================================
        % B) B2R2 + ML
        %% =====================================================
        delta_true = x_samp - x_mod;
        M = M_lambda(delta_true, lambda, length(x_samp));

        X_ft_lambda = fft(ifftshift(x_mod));

        x_rec = BBRR_matlab_real_signal( ...
            x_mod, X_ft_lambda, M, OF, lambda, length(x_samp));
        %% ================= MSE & PERFECT RECON INDICATOR =================
        % Calculate MSE between reconstructed and original (pre-modulo) samples
        %% ================= MSE & FAILURE ANALYSIS =================
       %% ================= PERFECT RECONSTRUCTION CHECK =================

        err = x_samp - x_rec;
        
        Energy = norm(x_samp)^2;
        
        mse = (norm(err)^2) / Energy;
        
        if mse < mse_threshold
            good_recovery(of_idx) = good_recovery(of_idx) + 1;
        end
               
        %% =================================================================
        % Rectify to symbol locations
        Tnyq = 1/(2*W);
        NoS = length(CoS_true);

        x_rect = zeros(NoS,1);
        for m = 1:NoS
            t_target = (2*m - NoS)*Tnyq;
            [~, idx2] = min(abs(t_s - t_target));
            x_rect(m) = x_rec(idx2);
        end

        dB = sum((A - x_rect).^2,1);
        [~, idxB] = min(dB);

        estB = A(:, idxB);
        SER_B2R2(of_idx, mc) = ...
            symbol_error_function(CoS_true(:), estB(:));

        if mod(mc,10)==0
            fprintf("  MC %d / %d\n", mc, num_MC);
            
        end
        
    end
    fprintf("Perfect B2R2 Recoveries = %d / %d\n", ...
        good_recovery(of_idx), num_MC);

    fprintf("Recovery Probability = %.3f\n\n", ...
        good_recovery(of_idx)/num_MC);
    %% ================= PLOT UPDATE ==========================
    meanA = mean(SER_ML(1:of_idx,:),2);
    meanB = mean(SER_B2R2(1:of_idx,:),2);
    meanC= mean(SER_no_fold(1:of_idx,:),2);
    plot(OF_list(1:of_idx), meanA,'-or','LineWidth',1.5);
    plot(OF_list(1:of_idx), meanB,'-sb','LineWidth',1.5);
    plot(OF_list(1:of_idx), meanC, '-og', ...
     'LineWidth', 1.5, ...
     'MarkerFaceColor','g');    
    legend("Exhaustive ML (Folded)", ...
       "B2R2 + ML", ...
       "No-Fold ML");    drawnow;
end
figure
plot(OF_list, good_recovery/num_MC,'-ok','LineWidth',2)
xlabel("Oversampling Factor")
ylabel("Perfect Reconstruction Probability")
title("B2R2 Perfect Recovery Probability")
grid on
fprintf("\nSimulation Complete.\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUPPORT FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B2R2 IMPLEMENTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x_rec = BBRR_matlab_real_signal(x_lambda, ~, M, OF, Lambda, L)

    x_lambda = x_lambda(:);
    rho = 1/OF;

    N = floor(M/2);
    beta = 1.02;
    decay = 0.999;
    reg = 1e-6;
    momentum = 0.6;
    epsilon = 1e-3*Lambda;

    w = linspace(0,2*pi,L);

    [~, idx1] = min(abs(w - beta*rho*pi));
    [~, idx2] = min(abs(w - (2*pi - beta*rho*pi)));

    mask = ones(L,1);
    mask(1:idx1+1)=0;
    mask(idx2:end)=0;

    D = diag(mask);
    F = DFT_matrix(L);
    dev = (D*F)'*(D*F);

    if OF==2
        mu = 2/L;
    else
        mu = 2/(idx2-idx1);
    end

    mom = momentum;
    change = zeros(L,1);

    delta = high_pass(-x_lambda, beta, rho, w);

    center = floor(L/2)+1;
    left  = max(1,center-N);
    right = min(L,center+N);

    delta(1:left-1)=0;
    delta(right+1:end)=0;

    d1=0; d2=0;

    for iter=1:5*10e4

        vec = x_lambda + delta;
        grad = dev*vec + reg*delta;

        change = mom*change + mu*real(grad);
        delta = delta - change;

        delta(1:left-1)=0;
        delta(right+1:end)=0;

        mu = decay*mu;
        mom = decay*mom;

        if mod(iter,5)==0

            d1n = delta(left);
            d2n = delta(right);

            if max(abs(d1n-d1),abs(d2n-d2)) < epsilon

                delta = round(delta/(2*Lambda))*(2*Lambda);

                if N<1
                    x_lambda(center) = x_lambda(center) + delta(center);
                    break;
                end

                x_lambda(left)  = x_lambda(left)  + delta(left);
                x_lambda(right) = x_lambda(right) + delta(right);

                N=N-1;
                left  = max(1,center-N);
                right = min(L,center+N);

                d1 = delta(left);
                d2 = delta(right);

                mu = 2/(idx2-idx1);
                mom = momentum;
            else
                d1=d1n; d2=d2n;
            end
        end
    end

    x_rec = x_lambda;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W = DFT_matrix(N)
    [i,j]=meshgrid(0:N-1,0:N-1);
    omega = exp(-2*pi*1i/N);
    W = omega.^((i - N/2).*j);
end

function y = high_pass(x, beta, rho, w)
    X = fft(ifftshift(x));
    [~,i1] = min(abs(w - beta*rho*pi));
    [~,i2] = min(abs(w - (2*pi - beta*rho*pi)));
    X(1:i1)=0;
    X(i2+1:end)=0;
    y = real(ifftshift(ifft(X)));
end