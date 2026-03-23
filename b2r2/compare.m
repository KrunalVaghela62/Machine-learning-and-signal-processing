addpath('C:\Users\Krunal\OneDrive\Documents\MATLAB\new_algo');
addpath('C:\Users\Krunal\OneDrive\Documents\MATLAB\linear_prediction');
function monte_carlo_simulation_gpu()
    % Parameters
    E = 10;
    lambda = 0.125;
    Delta = 2 * lambda;
    W = 50;
    NoS = 10;  % Number of sinc components
    beta = 1.05;
    
    % Oversampling factors
    OFs_part1=[2 2.5 3 3.5];
    OFs = unique([OFs_part1]);
    num_trials = 10;
    
    % Preallocate results on GPU
    mse_db1_all = gpuArray.zeros(num_trials, length(OFs));
    mse_db2_all = gpuArray.zeros(num_trials, length(OFs));
    mse_db3_all = gpuArray.zeros(num_trials, length(OFs)); % B2R2 method

    for idx_of = 1:length(OFs)
        OF = OFs(idx_of);
        fprintf('Running simulations for Oversampling Factor = %.3f (%d of %d)\n', OF, idx_of, length(OFs));
        for trial = 1:num_trials
            %% --- Generate random coefficients ONCE per trial ---
            coeffs = 0.5 * randn(NoS, 1);

            %% --- Generate continuous signal for algorithms 1 and 2 ---
            [x_orig_cont, t_d, ~] = generate_original_signal(E, W, OF, coeffs);

            %% --- Generate sampled signal ---
            [x_orig, t_s] = generate_sampled_signal(x_orig_cont, t_d); % implement this for your sampling system
            L = length(t_s); % match signal length

            % Move to GPU
            x_orig = gpuArray(x_orig);
            t_s = gpuArray(t_s);
            x_orig_cont = gpuArray(x_orig_cont);
            t_d = gpuArray(t_d);

            %% --- Modulo operation on GPU ---
            x_mod = modulo_operation(x_orig, Delta); % implement this

            Ts = t_s(2) - t_s(1);

            %% --- Algorithm 1: Corrected Recovery ---
            [x_rec1, success1] = modulo_recovery_corrected(x_mod, Ts, W, Delta, E, x_orig); % implement this
            x_rec1_cont = sinc_interpolation(x_rec1, t_s, t_d, Ts, OF); % implement this
            mse1_cont = (norm(x_orig_cont - x_rec1_cont))^2 / (norm(x_orig_cont))^2;
            mse_db1_all(trial, idx_of) = 10 * log10(mse1_cont);

            %% --- Algorithm 2: Next Sample Recovery ---
            K = compute_required_filter_length(W, E, Delta, Ts); % implement
            h_coeffs = design_chebyshev_predictor(K, W, Ts); % implement
            N_init = N_lambda(Delta, gather(x_orig)); % implement this if required on CPU
            c_n = possible_z_values(1, Delta); % implement

            z = gpuArray.zeros(length(x_orig), 1);
            x_rec2 = gpuArray.zeros(length(x_orig), 1);
            x_rec2(1:N_init-1) = x_mod(1:N_init-1);

            for n = N_init:length(x_mod)
                x_mod_h1 = x_mod_h(x_mod, h_coeffs, n); % implement
                z(n) = z_recovery(c_n, Delta, x_mod_h1, n, z, h_coeffs); % implement
                x_rec2(n) = x_mod(n) - z(n);
            end

            x_rec2_cont = sinc_interpolation(x_rec2, t_s, t_d, Ts, OF); % implement
            mse2_cont = (norm(x_orig_cont - x_rec2_cont))^2 / (norm(x_orig_cont))^2;
            mse_db2_all(trial, idx_of) = 10 * log10(mse2_cont);

            %% --- Algorithm 3: B2R2 Recovery (BBRR) ---
            % Set up domain matching sampled time axis
            n = -L/2 : L/2-1; % symmetric integer indices
            of = OF;           % Oversampling factor
            rho = 1 / of;
            Lambda = lambda;
            [x_n, ~, ~, ~] = generate_BL_signal(NoS, rho, n, of, coeffs);
            x_lambda = mod(real(x_n) + Lambda, 2*Lambda) - Lambda;
            X_ft_lambda = fft(ifftshift(x_lambda));
            delta1 = x_n - x_lambda;
            M = M_lambda(delta1, Lambda, L); % implement this

            % Use consistent bandwidth and freq/freq grid
            Ws = 2 * W * of;
            w = linspace(0, 2*pi*Ws, L);

            r_m_pgd = reconstruction_method(x_lambda, X_ft_lambda, M, of, w, W); % implement class
            x_rec3 = r_m_pgd.BBRR(Lambda, L);
            [t_dense3, x_dense3] = sinc_interpolate_bandlimited(x_n, n, beta, W, of); % implement
            [~, x_dense_rec3] = sinc_interpolate_bandlimited(x_rec3, n, beta, W, of);
            error3 = x_dense3 - x_dense_rec3;
            mse3_cont = (norm(error3))^2 / (norm(x_dense3))^2;
            mse_db3_all(trial, idx_of) = 10 * log10(mse3_cont);
        end
    end

    %% --- Transfer results back to CPU for plotting ---
    mse_db1_all = gather(mse_db1_all);
    mse_db2_all = gather(mse_db2_all);
    mse_db3_all = gather(mse_db3_all);

    %% --- Compute statistics ---
    mean_mse_db1 = mean(mse_db1_all, 1);
    std_mse_db1 = std(mse_db1_all, 0, 1);

    mean_mse_db2 = mean(mse_db2_all, 1);
    std_mse_db2 = std(mse_db2_all, 0, 1);

    mean_mse_db3 = mean(mse_db3_all, 1);
    std_mse_db3 = std(mse_db3_all, 0, 1);

    %% --- Plotting ---
    figure; hold on;
    errorbar(OFs, mean_mse_db1, std_mse_db1, '-ob', 'LineWidth', 1.5, 'DisplayName', 'Corrected Recovery');
    errorbar(OFs, mean_mse_db2, std_mse_db2, '-sr', 'LineWidth', 1.5, 'DisplayName', 'Next Sample Recovery');
    errorbar(OFs, mean_mse_db3, std_mse_db3, '-dg', 'LineWidth', 1.5, 'DisplayName', 'B2R2 Recovery');
    xlabel('Oversampling Factor (OF)');
    ylabel('MSE (dB)');
    title('Monte Carlo Simulation of Modulo Recovery Algorithms (GPU Accelerated)');
    legend('Location', 'best');
    grid on;
    hold off;
end
monte_carlo_simulation_gpu()
%% --- Signal Generation: Shared Coefficients interface ---
function [x_orig_cont, t_d, coeffs] = generate_original_signal(E, W, OF, coeffs)
    NoS = 10;
    BW = 2 * pi * W;
    Wnyq = 2 * BW;
    Ws = OF * Wnyq;
    Ts = 2 * pi / Ws;
    Td = Ts / 1000;
    t = -1 : Td : 1;
    t = t(:);
    if nargin < 4 || isempty(coeffs)
        coeffs = 0.5 * randn(NoS, 1);
    end
    f = zeros(length(t), 1);
    for m = 1 : NoS
        f = f + coeffs(m) * sinc(BW / pi * t - (m - NoS/2)*5);
    end
    c = 1;
    f = c * f / max(abs(f));
    x_orig_cont = f;
    t_d = t;
end

function [x_n, X_ft, E_x_n, Liphscitz_c] = generate_BL_signal(num_of_coeff, rho, n, of, coeffs)
    if nargin < 5 || isempty(coeffs)
        coeffs = (rand(1, num_of_coeff) - 0.5) * 2;
    end
    x_n = zeros(1, length(n));
    for k = 1:num_of_coeff
        offset = (k - ceil(num_of_coeff/2));
        x_n = x_n + coeffs(k) * sinc(rho * (n) - offset*5);
    end
    max_norm_bl = max(abs(x_n));
    x_n = x_n / max_norm_bl;
    X_ft = fft(ifftshift(x_n));
    E_x_n = (1 / of) * norm(x_n)^2;
    Liphscitz_c = max(abs(diff(x_n)));
end

% --- Example Sinc Interpolation (implement if missing) ---
function [t_dense, x_dense] = sinc_interpolate_bandlimited(x_n, n, beta, Wm, of)
    Ts = 1 / (2 * of * Wm);
    t_dense = linspace(n(1) * Ts, n(end) * Ts, numel(n) * 100);
    S = sinc((t_dense(:) - Ts * n(:)') / Ts);
    x_dense = S * x_n(:);
end

% --- Helper to sample the continuous signal at discrete points ---
function [x_samp, t_s] = generate_sampled_signal(x_orig_cont, t_d)
    % Example: sample every Nth point.
    % Adjust sampling scheme to match your requirements
    N = 1000; % tuning parameter, adjust to match L and W
    idx = 1:N:length(t_d);
    t_s = t_d(idx);
    x_samp = x_orig_cont(idx);
end

% --- Placeholder implementations for missing steps ---
function x_mod = modulo_operation(x, Delta)
    x_mod = mod(x + Delta, 2*Delta) - Delta;
end

function [x_rec, success] = modulo_recovery_corrected(x_mod, Ts, W, Delta, E, x_orig)
    % Dummy implementation: just return input
    x_rec = x_mod;
    success = true;
end

function x_interp = sinc_interpolation(x, t_s, t_d, Ts, OF)
    S = sinc((t_d(:) - t_s(:)') / Ts);
    x_interp = S * x(:);
end

function K = compute_required_filter_length(W, E, Delta, Ts)
    K = 16; % Placeholder
end

function h = design_chebyshev_predictor(K, W, Ts)
    h = ones(K, 1)/K; % Placeholder
end

function N = N_lambda(Delta, x)
    N = 1; % Placeholder
end

function c_n = possible_z_values(start, Delta)
    c_n = zeros(1, 10); % Placeholder
end

function xh = x_mod_h(x_mod, h_coeffs, n)
    xh = 0; % Placeholder
end

function z_val = z_recovery(c_n, Delta, x_mod_h1, n, z, h_coeffs)
    z_val = 0; % Placeholder
end

function M = M_lambda(delta, Lambda, L)
    thr = max(0.01, 0.05 * max(abs(delta)));
    idx = find(abs(delta) > thr);
    if isempty(idx)
        M = 5;
    else
        m_min = idx(1);
        m_max = idx(end);
        M = 2 * max(m_max - 0.5*L, 0.5*L - m_min) + 1;
    end
    M = min(L, max(floor(M), 5));
end

function r = reconstruction_method(x_lambda, X_ft_lambda, M, of, w, W)
    r.BBRR = @(Lambda, L) x_lambda; % Dummy: Replace with your BBRR implementation or class call
end

