function compare_LP_vs_B2R2_error_vs_clambda()
    %% Shared parameters
    c = 1;
    lambda_values = [ 0.5, 0.25, 0.125, 0.0625 , 0.03125, 0.015625];  % Lambda values
    %lambda_values = [ 0.99, 0.25, 0.125, 0.0625 , 0.03125, 0.015625];  % Lambda values
    C_by_Lambda = c ./ lambda_values;
    num_trials = 20;
    
    %% LP setup
    E = 50;
    W = 50;
    OF = 2;

    %% B2R2 setup
    coeff_num = 10;
    beta = 1.05;
    L = 2^10;
    n = -L/2 : L/2-1;
    of = 4;
    rho = 1 / of;
    Wm = 20;

    mean_errors_LP = zeros(size(lambda_values));
    std_errors_LP = zeros(size(lambda_values));
    mean_errors_B2R2 = zeros(size(lambda_values));
    std_errors_B2R2 = zeros(size(lambda_values));

    for i = 1:length(lambda_values)
        lambda = lambda_values(i);
        Delta = 2 * lambda;

        mse_lp = zeros(1, num_trials);
        mse_b2r2 = zeros(1, num_trials);

        fprintf('Running for λ = %.4f   (c/λ = %.2f)\n', lambda, C_by_Lambda(i));

        for trial = 1:num_trials
            %% --- Linear Prediction Method ---
            [x_orig_cont, t_d] = generate_original(E, W, OF);
            [x_orig, t_s] = generate_sampled(x_orig_cont, t_d);
            Ts = t_s(2) - t_s(1);
            x_mod = arrayfun(@(x) modulo_operation(x, Delta), x_orig);
            [x_rec, success] = modulo_recovery_corrected(x_mod, Ts, W, Delta, E);
            if success
                x_rec_cont = sinc_interpolation(x_rec, t_s, t_d, Ts);
                err_lp = x_rec_cont - x_orig_cont;
                mse_lp(trial) = 10 * log10(mean(err_lp.^2));
            else
                mse_lp(trial) = NaN;
            end

            %% --- B2R2 Method ---
            [x_n, ~, ~, ~] = generate_BL_signal(coeff_num, L, rho, n, of);
            x_n = c * x_n / max(abs(x_n));
            [~, x_dense] = sinc_interpolate_bandlimited(x_n, n, beta, Wm, of);
            x_lambda = mod(real(x_n) + lambda, 2*lambda) - lambda;
            X_ft_lambda = fft(ifftshift(x_lambda));
            delta = x_n - x_lambda;
            M = M_lambda(delta, lambda, L);
            w = linspace(0, 2*pi*2*Wm*of, L);  % Ws = 2*Wm*of
            r_m_pgd = reconstruction_method(x_lambda, X_ft_lambda, M, of, w, Wm);
            x_rec_b2r2 = r_m_pgd.BBRR(lambda, L);
            [~, x_dense_rec] = sinc_interpolate_bandlimited(x_rec_b2r2, n, beta, Wm, of);
            error_b2r2 = x_dense - x_dense_rec;
            mse_b2r2(trial) = 10 * log10(norm(error_b2r2)^2);
        end

        mean_errors_LP(i) = mean(mse_lp, 'omitnan');
        std_errors_LP(i) = std(mse_lp, 'omitnan');
        mean_errors_B2R2(i) = mean(mse_b2r2, 'omitnan');
        std_errors_B2R2(i) = std(mse_b2r2, 'omitnan');
    end

    %% Final Plot
    fig = figure('Name', 'Error vs c/lambda for LP and B2R2', ...
                 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    hold on;

    % LP Error Bars
    errorbar(C_by_Lambda, mean_errors_LP, std_errors_LP, 'b-o', ...
             'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Linear Prediction');

    % B2R2 Error Bars
    errorbar(C_by_Lambda, mean_errors_B2R2, std_errors_B2R2, 'r-s', ...
             'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'B2R2');

    xlabel('$\frac{c}{\lambda}$', 'Interpreter', 'latex', 'FontSize', 24);
    ylabel('MSE (dB)', 'FontSize', 20);
    title('Comparison of Recovery Error vs $c/\lambda$', 'Interpreter', 'latex', 'FontSize', 24);
    legend('Location', 'best', 'FontSize', 14);
    grid on;
    set(gca, 'FontSize', 16);
end

function [x_recovered, success] = modulo_recovery_corrected(x_star, Ts, W, Delta, E)
% MODULO_RECOVERY_CORRECTED Recovers bandlimited signal from modulo samples
% Based on Romanov-Ordentlich paper implementation
%
% Inputs:
%   x_star - modulo-reduced samples
%   Ts     - sampling period (must be < 1/(2*W))
%   W      - signal bandwidth
%   Delta  - modulo threshold
%   E      - signal energy bound
%
% Outputs:
%   x_recovered - recovered original samples
%   success     - recovery success flag

    % Verify Nyquist condition
    if Ts > 1/(2*W)
        error('Sampling rate must exceed Nyquist rate: Ts < 1/(2*W)');
    end
    
    N = length(x_star);
    x_recovered = zeros(size(x_star));
    success = true;
    
    % Find initialization point N where |x_n| < Delta/2
    % Use conservative estimate - start from beginning for simplicity
    N_init = 20;
    
    % Initialize with unfolded samples (assuming they're small enough)
    for n = 1:N_init
        x_recovered(n) = x_star(n);
    end
    
    % Design Chebyshev-based prediction filter
    K =compute_required_filter_length(W, E, Delta, Ts);%compute_required_filter_length(W, E, Delta, Ts);
    filter_length = 2*K;  % As per equation (9) in the paper
    h_coeffs = design_chebyshev_predictor(K, W, Ts);
    
    % Sequential recovery using the prediction-correction scheme
    for n = (N_init + 1):N
        % Predict current sample from past samples
        x_pred = compute_prediction(x_recovered, n, h_coeffs, filter_length);
        
        % Compute modulo-reduced prediction error
        e_star = modulo_operation(x_star(n) - x_pred, Delta);
        
        % Check if prediction error is within bounds
        if abs(e_star) >= Delta/2
            warning('Recovery failed at sample %d: prediction error too large', n);
            success = false;
            break;
        end
        
        % Recover current sample: x_n = x_pred + e_star
        x_recovered(n) = x_pred + e_star;
    end
end

function h_coeffs = design_chebyshev_predictor(K, W, Ts)
% DESIGN_CHEBYSHEV_PREDICTOR Designs predictor using Chebyshev polynomials
% Implements equation (7) from the paper
    
    % Compute interval bounds for Chebyshev polynomial
    a = 2 * cos(2*pi*W*Ts);  % Lower bound
    b = 2;                   % Upper bound
    
    % Generate Chebyshev polynomial T^[a,b]_K
    cheb_poly = generate_chebyshev_polynomial(K, a, b);
    
    % Construct p_K(z) = z^K * T^[a,b]_K(z + z^(-1))
    % This gives us the monic polynomial as in equation (9)
    p_coeffs = construct_predictor_polynomial(cheb_poly, K);
    
    % Extract filter coefficients h_1, ..., h_{2K}
    % From p_K(z) = -h_{2K}z^{2K} - ... - h_1z + 1
    h_coeffs = -p_coeffs(1:end-1);  % Remove constant term
end

function T_poly = generate_chebyshev_polynomial(K, a, b)
% GENERATE_CHEBYSHEV_POLYNOMIAL Creates Chebyshev polynomial on interval [a,b]
    
    if K == 0
        T_poly = 1;
        return;
    end
    
    % Transform to standard interval [-1, 1]
    % y = 2/(b-a) * (x - (a+b)/2)
    scale_factor = 2 / (b - a);
    offset = (a + b) / 2;
    
    % Generate Chebyshev polynomial using recurrence relation
    T_prev_prev = 1;                    % T_0
    T_prev = [scale_factor, -scale_factor * offset];  % T_1 transformed
    
    if K == 1
        T_poly = T_prev;
        return;
    end
    
    % Recurrence: T_{k+1}(y) = 2y*T_k(y) - T_{k-1}(y)
    for k = 2:K
        % 2y * T_k(y)
        term1 = conv([2*scale_factor, -2*scale_factor*offset], T_prev);
        
        % Pad T_{k-1} to match dimensions
        if length(T_prev_prev) < length(term1)
            T_prev_prev = [zeros(1, length(term1) - length(T_prev_prev)), T_prev_prev];
        elseif length(term1) < length(T_prev_prev)
            term1 = [zeros(1, length(T_prev_prev) - length(term1)), term1];
        end
        
        % T_{k+1} = 2y*T_k - T_{k-1}
        T_current = term1 - T_prev_prev;
        T_prev_prev = T_prev;
        T_prev = T_current;
    end
    
    % Apply scaling factor from equation (6)
    scaling = 2 * ((b - a) / 4)^K;
    T_poly = scaling * T_prev;
end

function p_coeffs = construct_predictor_polynomial(cheb_coeffs, K)
% Builds p_K(z) = z^K * T_K(z + z^{-1})
% cheb_coeffs: coefficients of T_K(x), length K+1, ordered from highest degree to constant term
% The function reverses them internally to constant-first order

    % Reverse input coefficients to ascending order (constant term first)
    cheb_coeffs = fliplr(cheb_coeffs);

    % Initialize polynomial coefficients vector for degree 2K polynomial
    p_coeffs = zeros(1, 2*K + 1);
    
    for i = 0:K
        c = cheb_coeffs(i+1);  % MATLAB 1-based indexing
        if c == 0
            continue;
        end
        
        % Expand (z + z^{-1})^i
        term = expand_z_plus_z_inv_power(i);  % length 2i+1
        
        % Calculate shift amount: shift right by (K - i)
        shift = K - i;
        
        % Add c * term shifted by 'shift' to p_coeffs
        p_coeffs(shift + (1:length(term))) = p_coeffs(shift + (1:length(term))) + c * term;
    end
    
    % Flip coefficients to MATLAB polynomial order (highest degree first)
    p_coeffs = fliplr(p_coeffs);
end



function expanded = expand_z_plus_z_inv_power(n)
% Expands (z + z^{-1})^n as a vector of coefficients
% Length = 2n + 1, center at n+1 corresponds to z^0

    expanded = zeros(1, 2*n + 1);
    center = n + 1;  % index for z^0

    if n == 0
        expanded(center) = 1;
        return;
    end

    for k = 0:n
        coeff = nchoosek(n, k);
        power = n - 2*k;  % power of z
        idx = center + power;
        % Accumulate coefficients (safe practice)
        expanded(idx) = expanded(idx) + coeff;
    end
end
function K = compute_required_filter_length(W, E, Delta, Ts)
    numerator = log(sqrt(32*W*E) / Delta);
    denominator = log(2 / (1 - cos(2*pi*W*Ts)));
    K = ceil(numerator / denominator);
    K = max(K, 1);
end

function x_pred = compute_prediction(x_recovered, n, h_coeffs, filter_length)
    x_pred = 0;
    l=length(h_coeffs);
    for i = 1:min(l, n - 1)
        x_pred = x_pred + h_coeffs(l-i+1) * x_recovered(n - i);
    end
end


function result = modulo_operation(x, Delta)
% MODULO_OPERATION Implements modulo operation as defined in the paper
% x* = [x] mod Delta as the unique number in [-Delta/2, Delta/2)
    
    result = x - Delta * round(x / Delta);
    
    % Ensure result is in [-Delta/2, Delta/2)
    if result >= Delta/2
        result = result - Delta;
    elseif result < -Delta/2
        result = result + Delta;
    end
end





function [x_orig_cont,t_d]=generate_original(E, W,OF)
    % Parameter
    NoS = 10;        % Number of sinc components (set this as needed)
    BW = 2 * pi * W; % Bandwidth in rad/s

    % Derived parameters
    Wnyq = 2 * BW;         % Nyquist frequency
    Tnyq = 2 * pi / Wnyq;  % Nyquist sampling interval
    Ws = OF * Wnyq;        % Sampling frequency with oversampling
    Ts = 2 * pi / Ws;      % Corresponding sampling period
    Td = Ts / 1000;        % Dense time axis step for plotting

    CoS = 0.5 * randn(NoS, 1);  % Random coefficients for sinc components
    t = -1 : Td : 1;           % Time axis
    t = t(:);                  % Ensure column vector

    f = zeros(length(t), 1);   % Initialize signal

    % Construct the bandlimited signal as a sum of sinc functions
    for m = 1 : NoS
        f = f + CoS(m) * sinc(BW / pi * t - (m - NoS/2)*5);
    end

    % Normalize signal
    c = 1; %maximum value attained by signal
    f = c * f / max(abs(f));
    x_orig_cont=f;
    t_d=t;
end
%[x_orig_cont,t_d]=generate_original(E, W,OF);
function [x_orig,t_s]=generate_sampled(x_orig_cont,t_d)
    x_orig = downsample(x_orig_cont, 1000); % Sample values
    t_s = downsample(t_d, 1000); % Sampling locations
end

function x_rec_cont = sinc_interpolation(x_rec, t_s, t_d, Ts)
% SINC_INTERPOLATION Reconstructs a continuous signal from samples using sinc interpolation
% Inputs:
%   x_rec - Recovered discrete samples
%   t_s   - Sampling times (e.g., n*Ts)
%   t_d   - Dense time vector for reconstruction
%   Ts    - Sampling period
% Output:
%   x_rec_cont - Interpolated continuous signal

    x_rec_cont = zeros(size(t_d));
    
    for n = 1:length(x_rec)
        x_rec_cont = x_rec_cont + x_rec(n) * sinc((t_d - t_s(n)) / Ts);
    end
end
function [x_n, X_ft, E_x_n, Liphscitz_c] = generate_BL_signal(num_of_coeff,L, rho, n, of)
    coeff = (rand(1, num_of_coeff) - 0.5) * 2;
    x_n = zeros(1, length(n));
    for k = 1:num_of_coeff
        offset = (k - ceil(num_of_coeff/2));
        x_n = x_n + coeff(k) * sinc(rho * (n - offset));
    end
    max_norm_bl = max(abs(x_n));
    x_n = x_n / max_norm_bl;
    X_ft = fft(ifftshift(x_n));
    E_x_n = (1 / of) * norm(x_n)^2;
    Liphscitz_c = max(abs(diff(x_n)));
end

function [t_dense, x_dense] = sinc_interpolate_bandlimited(x_n, n, beta,Wm,of)
    Ts = 1 / (2*of*Wm);  % Sampling period based on bandwidth

    t_dense = linspace(n(1)*Ts, n(end)*Ts, numel(n)*100);

    % Sinc interpolation matrix
    S = sinc((t_dense(:) - Ts * n(:)') / Ts);

    x_dense = S * x_n(:);
end


function M = M_lambda(delta, Lambda, L)
    thr = max(0.01, 0.05 * max(abs(delta)));  % More adaptive threshold
    idx = find(abs(delta) > thr);
    if isempty(idx)
        M = 5;  % Minimum window size even when no folding
    else
        m_min = idx(1);
        m_max = idx(end);
        M = 2 * max(m_max - 0.5*L, 0.5*L - m_min) + 1;
    end
    M = min(L, max(floor(M), 5));  % Always at least 5
end



function plot_signal(x_1, x_2, name_1, name_2, Lambda, Length, scale, n)
    % Find indices of n within the desired range
    center_time = 0; % Center at zero (or use mean(n) if not symmetric)
    idx = find(n >= center_time - scale & n <= center_time + scale);

    plot(n(idx), x_1(idx), '-r'); hold on;
    plot(n(idx), x_2(idx), 'b--', 'LineWidth', 1);
    xlim([center_time - scale, center_time + scale]);
    yline(0, 'k-', 'LineWidth', 0.75);
    yline(Lambda, 'Color', [0 0 0.5], 'LineStyle', '-', 'LineWidth', 0.25);
    yline(-Lambda, 'Color', [0 0 0.5], 'LineStyle', '-', 'LineWidth', 0.25);
    text(center_time-scale-10, Lambda, '\lambda', 'FontSize', 16, 'BackgroundColor', 'w');
    text(center_time-scale-12, -Lambda, '-\lambda', 'FontSize', 16, 'BackgroundColor', 'w');
    legend({name_1, name_2}, 'Location', 'northeast');
    set(gca, 'XTick', [], 'YTick', []);
    hold off;
    drawnow;
end


