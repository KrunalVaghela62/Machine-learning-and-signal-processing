%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Monte-Carlo SER (Peak Sampling Only)
% No OF iteration
% H = Identity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% ================= PARAMETERS =========================
num_MC = 2;

S  = [-1 1 2 -2];
E  = 10;
W  = 10;

lambda = 0.2;
Delta  = 2*lambda;
sigma  = lambda*2;

rng(0);

SER_unfold = zeros(num_MC,1);
SER_fold   = zeros(num_MC,1);

%% --- Generate once to get dimensions ---
[x_cont_tmp, t_d, ~, NoS] = ...
    generate_analog_transmit(E, W, 1, S);

% Build codebook once
A = generate_all_as(S, NoS);     % size: NoS × NumComb
NumComb = size(A,2);

fprintf("\nRunning Monte-Carlo...\n");

%% ================= MONTE CARLO ==========================
for mc = 1:num_MC

    %% 1) Generate signal
    [x_cont, t_d, CoS_true, ~] = ...
        generate_analog_transmit(E, W, 1, S);
    CoS_true
    %% 2) Add noise (continuous domain)
    noise = sigma * randn(length(t_d),1);

    Td = t_d(2) - t_d(1);
    Fs = 1/Td;

    noise = lowpass(noise, W, Fs, ...
        ImpulseResponse="iir", Steepness=0.9999);

    x_noisy_cont = x_cont + noise;

    %% 3) Sample at peaks
    [x_samp, t_s] = ...
        sample_at_sinc_peaks(x_noisy_cont, t_d, W, NoS);

    x_samp = x_samp(:)

    %% =====================================================
    % A) UNFOLDED ML
    %% =====================================================
    diff_unfold = A - x_samp;
    d_unfold = sum(diff_unfold.^2, 1);
    [~, idxU] = min(d_unfold);

    estU = A(:, idxU);

    SER_unfold(mc) = ...
        symbol_error_function(CoS_true(:), estU(:));

    %% =====================================================
    % B) FOLDED ML
    %% =====================================================
    x_mod = mod(x_samp + lambda, 2*lambda) - lambda
    A_mod = mod(A + lambda, 2*lambda) - lambda;

    diff_fold = A_mod - x_mod;
    d_fold = sum(diff_fold.^2, 1);
    [~, idxF] = min(d_fold);

    estF = A(:, idxF);

    SER_fold(mc) = ...
        symbol_error_function(CoS_true(:), estF(:));

    if mod(mc,20)==0
        fprintf("  MC %d / %d\n", mc, num_MC);
    end
end

%% ================= RESULTS ================================
mean_SER_unfold = mean(SER_unfold);
mean_SER_fold   = mean(SER_fold);

fprintf("\n=============================\n");
fprintf("Unfolded SER = %.4f %%\n", mean_SER_unfold);
fprintf("Folded   SER = %.4f %%\n", mean_SER_fold);
fprintf("=============================\n");
function [x_orig, t_s] = sample_at_sinc_peaks(x_orig_cont, t_d, W, NoS)

    % Nyquist spacing
    Tnyq = 1/(2*W);

    % Compute peak locations
    t_s = zeros(NoS,1);
    for m = 1:NoS
        t_s(m) = (2*m - NoS) * Tnyq;
    end

    % Interpolate from dense grid
    x_orig = interp1(t_d, x_orig_cont, t_s, 'linear');

end