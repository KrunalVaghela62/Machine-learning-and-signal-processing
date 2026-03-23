%{
function montecarlo_error_vs_OF
    % Parameters
    Lambda = 0.25;
    N_trials = 50;
    Wm=20;
    OF_values = 2:8;
    beta = 1.05;
    coeff_num = 10;
    scale = 100;
    L = 2^10;
    n = -L/2 : L/2-1;
    rho = 1 ./ OF_values;
    mse_db = zeros(length(OF_values), N_trials);

    for i = 1:length(OF_values)
        of = OF_values(i);
        for trial = 1:N_trials
            [x_n, ~, Energy, ~] = generate_BL_signal(coeff_num, L, 1/of, n, of);

            % Interpolation
            [t_dense, x_dense] = sinc_interpolate_bandlimited(x_n, n, beta,Wm,of);
            x_lambda = mod(real(x_n) + Lambda, 2*Lambda) - Lambda;

            % Reconstruction
            X_ft_lambda = fft(ifftshift(x_lambda));
            delta1 = x_n - x_lambda;
            M = M_lambda(delta1, Lambda, L);
            Ws=2*Wm*of;
            r_m_pgd = reconstruction_method(x_lambda, X_ft_lambda, M, of, linspace(0,2*pi*Ws,L),Wm);
            x_rec = r_m_pgd.BBRR(Lambda, L);
            [~, x_dense_rec] = sinc_interpolate_bandlimited(x_rec, n, beta,Wm,of);

            % Error calculation
            error = x_dense - x_dense_rec;
            mse = (norm(error)^2);
            mse_db(i, trial) = 10 * log10(mse);
        end
        disp(['Completed OF = ', num2str(of)]);
    end

    % Plot
    figure;
    errorbar(OF_values, mean(mse_db, 2), std(mse_db, 0, 2), '-o', ...
             'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Oversampling Factor (OF)');
    ylabel('MSE (dB)');
    title('Error vs Oversampling Factor');
    grid on;
end
%}

function montecarlo_error_vs_clambda
    %% Parameters
    c = 1;  % Peak amplitude (fixed)
    lambda_values = [0.99,0.5,0.25,0.125,1/16];
    clambda_values = c ./ lambda_values;
    Wm=20;
    N_trials = 30;
    beta = 1.05;
    coeff_num = 10;
    L = 2^10;
    n = -L/2 : L/2-1;
    of = 4;
    rho = 1 / of;

    mse_db = zeros(length(lambda_values), N_trials);

    %% Monte Carlo Simulation
    for i = 1:length(lambda_values)
        lambda = lambda_values(i);
        for trial = 1:N_trials
            % Generate bandlimited signal
            [x_n, ~, Energy, ~] = generate_BL_signal(coeff_num, L, rho, n, of);
            x_n = c * x_n / max(abs(x_n));  % Scale to peak c
            Ws=2*Wm*of;
            w = linspace(0, 2*pi*Ws, L);

            % Densely interpolate original
            [~, x_dense] = sinc_interpolate_bandlimited(x_n, n, beta,Wm,of);

            % Apply modulo operation
            x_lambda = mod(real(x_n) + lambda, 2*lambda) - lambda;

            % Folded DTFT
            X_ft_lambda = fft(ifftshift(x_lambda));
            delta = x_n - x_lambda;
            M = M_lambda(delta, lambda, L);

            % Reconstruct
            r_m_pgd = reconstruction_method(x_lambda, X_ft_lambda, M, of, w,Wm);
            x_rec = r_m_pgd.BBRR(lambda, L);

            % Interpolate recovered signal
            [~, x_dense_rec] = sinc_interpolate_bandlimited(x_rec, n, beta,Wm,of);

            % Compute error
            error = x_n - x_rec;
            mse = norm(error)^2;
            mse_db(i, trial) = 10 * log10(mse);
        end
        disp(['Completed λ = ', num2str(lambda), '  (c/λ = ', num2str(clambda_values(i)), ')']);
    end

    %% Plot Error vs c/lambda
    figure;
    errorbar(clambda_values, mean(mse_db, 2), std(mse_db, 0, 2), '-o', ...
             'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('c / \lambda');
    ylabel('MSE (dB)');
    title('Monte Carlo: Error vs c / \lambda');
    grid on;
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


