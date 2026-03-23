function main
    % Parameters
    L = 2^10;
    of =2;
    Wm=50;
    Ws=2*Wm*of;
    w = linspace(0, 2*pi*Ws, L);
    n = -L/2 : L/2-1;
    beta = 1.05;
    rho = 1 / of;
    Lambda = 1/8;
    coeff_num = 10;
    scale = 100;
    upper_sampling_fac=2;
    
    % Generate BL signal
    %[x_n, X_ft, Energy, Liphscitz_c] = generate_BL_signal(coeff_num, rho, n, of);

    [x_n, X_ft , Energy, ~] = generate_BL_signal(coeff_num, rho, n, of);
    [t_dense , x_dense] = sinc_interpolate_bandlimited(x_n, n, beta,Wm,of);
    x_lambda = mod(real(x_n) + Lambda, 2*Lambda) - Lambda;
    X_ft_lambda = fft(ifftshift(x_lambda));
    delta1 = x_n - x_lambda;
    M = M_lambda(delta1, Lambda, L);

    disp(['Number of folded samples = ', num2str(M)]);
    plot_signal(x_lambda, x_n, 'Modulo', 'Original', Lambda, L, scale, n);

    pause(2);

    % Reconstruction
    tic;
    r_m_pgd = reconstruction_method(x_lambda, X_ft_lambda, M, of, w,Wm);
    x_rec = r_m_pgd.BBRR(Lambda, L);
    toc_val = toc;
    disp(size(x_rec));
    disp(['Inference Time = ', num2str(toc_val)]);
    [t_dense_rec, x_dense_rec] = sinc_interpolate_bandlimited(x_rec, n, beta,Wm,of);

    % Error
    error = x_dense - x_dense_rec;
    mse = (norm(error))^2 ;
    disp(['MSE = ', num2str(mse), ' , MSE(dB) = ', num2str(10*log10(mse))]);

    plot_signal(x_rec, x_n, 'Recovery', 'Original', Lambda, L, scale, n);
end
function [x_n, X_ft, E_x_n, Liphscitz_c] = generate_BL_signal(num_of_coeff, rho, n, of)
    coeff = (rand(1, num_of_coeff) - 0.5) * 2;
    x_n = zeros(1, length(n));
    for k = 1:num_of_coeff
        offset = (k - ceil(num_of_coeff/2));
        x_n = x_n + coeff(k) * sinc(rho * (n) - offset*5);
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
    center_time = 0;
    idx = find(n >= center_time - scale & n <= center_time + scale);

    % Use plot for continuous lines with thickness for 'curviness'
    plot(n(idx), x_1(idx), 'b', 'LineWidth', 4); hold on;
    plot(n(idx), x_2(idx), 'r--', 'LineWidth', 3);

    % Optional: stem for discrete samples (uncomment if needed)
    % stem(n(idx), x_1(idx), 'b', 'LineWidth', 2);
    % stem(n(idx), x_2(idx), 'r', 'LineWidth', 2);

    xlim([center_time - scale, center_time + scale]);
    yline(0, 'k-', 'LineWidth', 0.75);
    yline(Lambda, 'Color', [0 0 0.5], 'LineStyle', '-', 'LineWidth', 0.25);
    yline(-Lambda, 'Color', [0 0 0.5], 'LineStyle', '-', 'LineWidth', 0.25);
    text(center_time-scale-10, Lambda, '\lambda', 'FontSize', 16, 'BackgroundColor', 'w');
    text(center_time-scale-12, -Lambda, '-\lambda', 'FontSize', 16, 'BackgroundColor', 'w');
    legend({name_1, name_2}, 'Location', 'northeast');
    set(gca,'FontSize',30,'FontName','Times')
    title(['c/$\lambda = ', num2str(1/Lambda), '$'], 'Interpreter', 'latex', 'FontSize', 18);
    hold off;
    drawnow;
end

