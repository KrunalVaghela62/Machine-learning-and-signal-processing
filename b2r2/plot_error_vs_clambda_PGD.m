function plot_error_vs_clambda_PGD()
    % Setup
    L = 2^10;
    w = linspace(0, 2*pi, L);
    n = -L/2 : L/2-1;
    of = 2;
    rho = 1 / of;
    coeff_num = 102;
    num_trials = 20;
    lambda_list = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1];
    c = 1;  % Max amplitude of normalized signal
    clambda_ratio = c ./ lambda_list;

    mean_mse_db = zeros(size(lambda_list));
    std_mse_db = zeros(size(lambda_list));

    for i = 1:length(lambda_list)
        Lambda = lambda_list(i);
        mse_db_all = zeros(1, num_trials);

        for trial = 1:num_trials
            % Generate bandlimited signal
            fprintf('  Trial %d/%d for c/Lambda = %.2f\n', trial, num_trials, clambda_ratio(i));

            [x_n, X_ft, Energy, ~] = generate_BL_signal(coeff_num, L, rho, n, of);
            [~, x_dense] = sinc_interpolate_bandlimited( x_n, n, 5);  % fixed upsampling

            % Modulo folding
            x_lambda = mod(real(x_n) + Lambda, 2*Lambda) - Lambda;
            X_ft_lambda = fft(ifftshift(x_lambda));
            delta1 = x_n - x_lambda;
            M = M_lambda(delta1, Lambda, L);

            try
                % Reconstruction
                r_m_pgd = reconstruction_method(x_lambda, X_ft_lambda, M, of, w);
                x_rec = r_m_pgd.BBRR(Lambda, L);
                [~, x_dense_rec] = sinc_interpolate_bandlimited( x_rec, n, 5);

                % Error calculation
                error = x_dense - x_dense_rec;
                mse = (norm(error))^2 / Energy;
                mse_db = 10 * log10(mse);

                if isfinite(mse_db)
                    mse_db_all(trial) = mse_db;
                else
                    mse_db_all(trial) = NaN;
                end
            catch
                mse_db_all(trial) = NaN;
            end
        end

        % Filter out NaNs
        valid = isfinite(mse_db_all);
        if any(valid)
            mean_mse_db(i) = mean(mse_db_all(valid));
            std_mse_db(i) = std(mse_db_all(valid));
        else
            mean_mse_db(i) = NaN;
            std_mse_db(i) = NaN;
        end

        fprintf('c/Lambda = %.2f done\n', clambda_ratio(i));
    end

    % Remove invalid entries before plotting
    valid_plot = isfinite(mean_mse_db) & isfinite(std_mse_db);
    clambda_ratio = clambda_ratio(valid_plot);
    mean_mse_db = mean_mse_db(valid_plot);
    std_mse_db = std_mse_db(valid_plot);

    % Plot
    figure('Name', 'PGD Recovery Error vs c/\lambda', 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    hold on;
    fill([clambda_ratio fliplr(clambda_ratio)], ...
         [mean_mse_db - std_mse_db, fliplr(mean_mse_db + std_mse_db)], ...
         [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    plot(clambda_ratio, mean_mse_db, '-o', 'LineWidth', 2, 'Color', 'b');
    xlabel('$\frac{c}{\lambda}$', 'Interpreter', 'latex', 'FontSize', 20);
    ylabel('MSE (dB)', 'FontSize', 20);
    title('PGD Recovery Error vs $\frac{c}{\lambda}$', 'Interpreter', 'latex', 'FontSize', 22);
    legend('Mean ± Std Dev', 'Mean MSE', 'Location', 'best');
    grid on;
    set(gca, 'FontSize', 14);
end
function [x_n, X_ft, E_x_n, Liphscitz_c] = generate_BL_signal(num_of_coeff, L, rho, n, of)
    coeff = (rand(1, num_of_coeff) - 0.5) * 2;
    x_n = zeros(1, L);
    for k = 1:num_of_coeff
        offset = (k - ceil(num_of_coeff/2)) * 15;
        x_n = x_n + coeff(k) * sinc(rho * (n - offset));
    end
    max_norm_bl = max(abs(x_n));
    x_n = x_n / max_norm_bl;
    X_ft = fft(ifftshift(x_n));
    E_x_n = (1 / of) * norm(x_n)^2;
    Liphscitz_c = max(abs(diff(x_n)));
end

function [t_dense, x_dense] = sinc_interpolate_bandlimited(x_n, n, upsample_factor)
    Ts = n(2) - n(1);
    t_dense = linspace(n(1), n(end), length(n)* upsample_factor);
    S = sinc((t_dense(:) - n(:)') / Ts);
    x_dense = S * x_n(:);
end

function M = M_lambda(delta, Lambda, L)
    thr = 0.1 * Lambda;
    idx = find(abs(delta) > thr);
    if isempty(idx)
        M = 1;
        return;
    end
    m_min = idx(1);
    m_max = idx(end);
    M = 2 * max(m_max - 0.5 * L, 0.5 * L - m_min) + 1;
    M = floor(M);
end