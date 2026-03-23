% run_b2r2_analog_pipeline.m
% Wires: generate_analog_transmit -> sample (generate_sampled_signal) -> modulo -> B2R2 reconstruction
% Implementation uses FFT-based partial-DTFT operators (no huge dense matrices)
%
% Requirements: place your generate_analog_transmit and generate_sampled_signal on path
% (or keep the provided versions at end of this file) and run this script.

clear; close all; clc;

%% ========================= PARAMETERS (top block) =========================
OF_analog = 4;        % desired oversampling factor for analog sampling (Fs = OF_analog * 2 * W)
W = 10;               % analog bandwidth (Hz) used by generate_analog_transmit
E = 10;               % unused but kept for generator signature
S = [-1 1];           % coefficient set for generator
Lambda = 0.25;        % modulo threshold
SNR_dB = 40;          % SNR added to continuous signal (dB)

% B2R2 internal params (you can tune these)
max_pgd_iters = 2000;    % PGD iters per stage
pgd_tol = 1e-6;          % PGD convergence tolerance (change in z)
gamma = 0.8;             % PGD step-size scaling (tuned empirically)
smooth_window = 5;       % smoothing used when estimating N_lambda
plot_window = 120;       % half-width for plotting around center
verbose = true;
%% ==========================================================================

%% 1) Generate analog continuous signal
[x_orig_cont, t_d, CoS, NoS] = generate_analog_transmit(E, W, OF_analog, S);

% Add AWGN to continuous signal (measured SNR)
x_cont_noisy = awgn(x_orig_cont, SNR_dB, 'measured');

% compute continuous time grid properties
Td = t_d(2) - t_d(1);
Fs_cont = 1 / Td;
duration = t_d(end) - t_d(1);

fprintf('Continuous signal: duration=%.4fs, dense Fs=%.2f Hz, samples=%d\n', duration, Fs_cont, length(t_d));
fprintf('Analog generator W=%.2f Hz, requested OF_analog=%.2f -> desired Fs = %.2f Hz\n', W, OF_analog, OF_analog*2*W);

%% 2) Sample the continuous-time (use your provided function: downsample by 1000)
[x_sampled, t_s] = generate_sampled_signal(x_cont_noisy, t_d);    % this downsamples by 1000 as you provided
x_sampled = x_sampled(:);
t_s = t_s(:);

% actual discrete sampling frequency (from t_s)
Td_s = t_s(2) - t_s(1);
Fs = 1 / Td_s;
L = length(x_sampled);
n = (-floor(L/2) : ceil(L/2)-1).';

fprintf('Discrete samples: Fs=%.3f Hz, L=%d\n', Fs, L);

%% 3) Compute discrete normalized signal bandwidth rho (digital fraction)
% digital radian freq of analog max = 2*pi*W/Fs
omega_m = 2*pi*W / Fs;
rho = omega_m / pi;       % so digital band is |omega| < rho*pi
fprintf('Derived digital rho = %.5f (omega_m = %.4f rad)\n', rho, omega_m);

% if your sampling Fs is not at OF_analog*2W, you might see mismatch. That's fine:
fprintf('Note: desired rho=1/OF_analog = %.5f\n', 1/OF_analog);

%% 4) Apply modulo operator (discrete) to sampled signal
x_true = real(x_sampled);                    % original discrete-time signal (noisy)
x_lambda = mod(x_true + Lambda, 2*Lambda) - Lambda;   % folded samples

%% 5) Estimate half-support N (N_lambda) from true residual (only for simulation / to set algorithm)
delta_true = x_true - x_lambda;
N_est = detect_half_support(delta_true, Lambda, smooth_window);
fprintf('Estimated half-support N_est = %d (full folded width M ~ %d)\n', N_est, 2*N_est+1);

%% 6) Run B2R2 reconstruction (my FFT-based PGD implementation)
% We'll implement the iterative/peeling strategy described in the paper:
% - initialize z0 = PS_{N}(ifft(mask .* FFT(x_lambda))) (projected inverse partial-DTFT)
% - run PGD minimizing || F_rho (z - x_lambda) ||^2 subject to support SN
% - round z to 2*Lambda multiples and subtract from x_lambda, shrink support, repeat

% build frequency mask (select region  rho*pi < |omega| <= pi )
w = linspace(0,2*pi,L);
mask = make_highband_mask(L, rho, 1.05);  % uses beta~1.05 to allow small margin (as in paper)

tic;
[x_rec, z_total] = B2R2_fft(x_lambda, Lambda, N_est, mask, max_pgd_iters, pgd_tol, gamma, verbose);
tocTime = toc;
fprintf('Reconstruction done in %.3f s\n', tocTime);

%% 7) Evaluate & plot
err = x_true - x_rec;
mse = sum(abs(err).^2) / sum(abs(x_true).^2);
fprintf('MSE = %.6e  (%.2f dB)\n', mse, 10*log10(mse));

% plotting center region
center = floor(L/2) + 1;
left = max(1, center - plot_window);
right = min(L, center + plot_window);
figure; plot(n(left:right), x_true(left:right), '-r','LineWidth',1.2); hold on;
plot(n(left:right), x_rec(left:right), '--b','LineWidth',1.1); legend('Original sampled','Recovered'); title('Original vs Recovered (center window)');
grid on;

figure; plot(n(left:right), x_lambda(left:right), '-k','LineWidth',1.0); hold on;
plot(n(left:right), x_true(left:right),'--r','LineWidth',1.0); legend('Modulo','Original'); title('Modulo vs Original (center window)');
grid on;

%% ========================= Functions ======================================

function mask = make_highband_mask(L, rho, beta)
    % mask: boolean vector length L selecting frequencies with |omega| > beta*rho*pi and
    % below Nyquist (i.e., outside the baseband where true signal lives)
    % w indices correspond to fftshifted frequencies when using fftshift; here we operate with unshifted FFT indexing.
    % We'll build mask in the standard FFT ordering (0..2pi).
    w = linspace(0, 2*pi, L);
    omega_cut_low = beta * rho * pi;
    omega_cut_high = 2*pi - omega_cut_low;
    mask = false(L,1);
    % select indices where w in (omega_cut_low, omega_cut_high) complement => we want the high-band outside |omega| <= beta*rho*pi
    % But according to the paper we want the out-of-band region rho*pi < |omega| < pi; this implementation uses beta margin
    mask( (w > omega_cut_low) & (w < omega_cut_high) ) = true;
    % Note: mask is true for frequencies used to identify residual z (F_rho).
end

function N = detect_half_support(delta, Lambda, smooth_w)
    % robust half-width detection from delta (true residual) - only available in simulation
    if all(abs(delta) < 1e-12)
        N = 0; return;
    end
    a = movmean(abs(delta), smooth_w);
    thr = 0.2 * Lambda;
    idx = find(a > thr);
    if isempty(idx)
        N = 0; return;
    end
    Ld = length(delta);
    center = floor(Ld/2) + 1;
    N = max(abs(idx(1) - center), abs(idx(end) - center));
end

function [x_rec, z_total] = B2R2_fft(x_lambda, Lambda, N_init, mask, max_iters, tol, gamma, verbose)
    % FFT-based B2R2 implementation
    % Inputs:
    %   x_lambda : folded samples (Lx1)
    %   Lambda   : modulo half-range
    %   N_init   : initial half-support (integer)
    %   mask     : boolean mask selecting out-of-band frequencies (length L)
    % Outputs:
    %   x_rec    : reconstructed (unfolded) samples
    %   z_total  : estimated residual (same length)
    %
    % Approach:
    %  - iterative peeling from outer edges inwards (reduce N by 1 each successful stage)
    %  - each stage: run PGD to estimate z on support S_N (center +/- N)
    %  - PGD uses gradient computed as ifft(mask .* fft(z - x_lambda)) (adjoint of partial DTFT)
    %  - after convergence quantize stage estimate to multiples of 2*Lambda, subtract and shrink support

    L = length(x_lambda);
    center = floor(L/2) + 1;
    z_total = zeros(L,1);          % accumulated residual
    x_work = x_lambda;             % progressively updated folded data (we subtract estimated residuals)
    N = N_init;

    % Precompute mask as column
    mask_vec = double(mask(:));

    stage = 0;
    while N >= 0
        stage = stage + 1;
        if verbose
            fprintf('Stage %d: estimating residual on support N = %d  (center indices %d:%d)\n', stage, N, center-N, center+N);
        end

        % initialize z0 = projection of inverse partial DTFT of x_work
        % compute partial inverse DTFT: z0_hat = ifft(mask .* FFT(x_work))
        z0 = real(ifft( mask_vec .* fft(x_work) ));
        % project onto support SN (zero outside)
        z0(1:center-N-1) = 0;       % left of support
        z0(center+N+1:end) = 0;     % right of support

        z = z0;
        change = Inf;
        iter = 0;

        % PGD loop (minimize || F_rho (z - x_work) ||^2 with z supported on SN)
        while (iter < max_iters) && (change > tol)
            iter = iter + 1;
            % compute gradient: grad = F_rho^*( F_rho (z - x_work) )
            res_freq = mask_vec .* fft(z - x_work);     % F_rho(z - x_work) in freq domain
            grad = real(ifft(res_freq));                % apply adjoint -> time domain gradient (no extra scaling)
            % step
            z_new = z - gamma * grad;
            % projection onto support SN
            z_new(1:center-N-1) = 0;
            z_new(center+N+1:end) = 0;
            change = norm(z_new - z) / (norm(z) + 1e-12);
            z = z_new;
            % small verbosity
            if (mod(iter,200)==0) && verbose
                fprintf('  PGD iter %d, rel change %.3e\n', iter, change);
            end
        end

        if verbose
            fprintf('  PGD finished after %d iters, rel change %.3e\n', iter, change);
        end

        % Quantize the current z estimate to integer multiples of 2*Lambda
        z_q = round(z / (2*Lambda)) * (2*Lambda);

        % If rounded z is (almost) zero, then nothing new -> break to avoid infinite loop
        if norm(z_q) < 1e-12
            if verbose
                fprintf('  Rounded z is (near) zero; stopping.\n');
            end
            break;
        end

        % Accumulate and subtract from working folded samples
        z_total = z_total + z_q;
        x_work = x_work - z_q;

        % shrink support and continue (peeling)
        if N <= 0
            break;
        else
            N = N - 1;
        end
    end

    % final recovered signal
    x_rec = x_work;
end

%% ========================= Provided generator functions =================
% If you already have these exact functions as files, comment the copies below
% and let MATLAB use the external ones. I include copies to make script self-contained.

function [x_orig_cont,t_d,CoS,NoS] = generate_analog_transmit(E, W, OF, S)
    % Provided by user (kept as-is)
    NoS = 5;                % Number of sinc components
    BW = 2 * pi * W;         % Bandwidth (rad/s)

    % Derived parameters
    Wnyq = 2 * BW;
    Tnyq = 2 * pi / Wnyq;    
    Ws   = OF * Wnyq;
    Ts   = 2 * pi / Ws;
    Td   = Ts / 1000;        % Dense time grid

    % Coefficients for sinc amplitudes
    CoS = S(randi(numel(S), NoS, 1));
    t = (-1 : Td : 1).';
    f = zeros(length(t),1);  % Sum of all sinc pulses

    for m = 1:NoS
        y_m = CoS(m) * sinc( 2*W * (t - ((2*m-(NoS))*Tnyq)));
        f = f + y_m;
    end
    x_orig_cont = f;
    t_d = t;
end

function [x_orig,t_s]=generate_sampled_signal(x_orig_cont,t_d)
    % Provided by user (kept as-is): downsample by 1000
    x_orig = downsample(x_orig_cont, 1000); % Sample values
    t_s = downsample(t_d, 1000); % Sampling locations
end
