clc; clear; close all;

%% ================= PARAMETERS =================
E  = 0.01;                                       
W  = 10;
OF = 2;
S  = [-2 -1 1 2];

Lambda = 0.2;              % TRUE modulo parameter
Delta  = 2*Lambda;

%% ================= SIGNAL GENERATION =================
[x_orig_cont, t_d, CoS, NoS] = generate_analog_transmit(E, W, OF, S);
[x_orig, t_s] = generate_sampled_signal(x_orig_cont, t_d);

x_orig = x_orig(:);
L = length(x_orig);

%% ================= FOLDING =================
x_lambda = mod(x_orig + Lambda, Delta) - Lambda;
z_true   = x_orig - x_lambda;

%% ================= SEARCH SPACE =================
c1 = 5;           % safe bound
[c_n, ~] = possible_z_values(c1, Delta);
Zc = c_n * Delta;

fprintf('Signal length L = %d\n', L);
fprintf('Search space size |Zc| = %d\n', length(Zc));

%% ================= INITIAL SUPPORT SIZE =================
N0 = M_lambda(z_true, Lambda, L);    % oracle for testing
fprintf('Initial support size N0 = %d\n', N0);

%% ================= B2R2 UNFOLDING =================
x_rec = B2R2_search_support_peeling( ...
            x_lambda, OF, Lambda, Zc, N0);

%% ================= METRICS =================
NMSE = norm(x_orig - x_rec)^2 / norm(x_orig)^2;
fprintf('Final NMSE = %.3e\n', NMSE);

%% ================= PLOTS =================
figure;
plot(t_s, x_orig, 'k', 'LineWidth', 1.5); hold on;
plot(t_s, x_lambda, 'r');
legend('x original','x folded');
title('Original vs Folded');

figure;
plot(t_s, z_true, 'b');
title('True residue');

figure;
plot(t_s, x_orig, 'k', 'LineWidth', 1.5); hold on;
plot(t_s, x_rec, 'r');
legend('x original','x reconstructed');
title('B2R2 (Boundary Search + Support Peeling)');

function [x_orig, t_s] = generate_sampled_signal(x_orig_cont, t_d)
    % Your downsampling by 1000 (as before)
    x_orig = downsample(x_orig_cont, 1000);
    t_s = downsample(t_d, 1000);
end
function [c_n,n]=possible_z_values(c,delta)%z=delta*c_n
    n = floor((c/delta)+0.5);
    c_n=-n:n;
end
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

function x_rec = B2R2_search_support_peeling(x_lambda, OF, Lambda, Zc, N0)

    x_lambda = x_lambda(:);
    L = length(x_lambda);
    center = floor(L/2) + 1;
    rho = 1/OF;
    beta = 1.05;

    %% ===== High-pass operator =====
    w = linspace(0,2*pi,L);
    [~, i1] = min(abs(w - beta*rho*pi));
    [~, i2] = min(abs(w - (2*pi - beta*rho*pi)));

    D = ones(L,1);
    D(1:i1) = 0;
    D(i2:end) = 0;
    D = diag(D);

    F = DFT_matrix(L);
    A = real(F' * D * F);

    %% ===== PGD parameters =====
    mu = 1/L;              % stable step size
    max_iter = 3000;
    tol = 1e-6;

    %% ===== Initialization =====
    z = high_pass_matlab(-x_lambda, beta, rho, w);

    %% ===== SUPPORT PEELING LOOP =====
    for N = N0:-1:0

        left  = center - N;
        right = center + N;

        %% ---- PGD WITH FIXED SUPPORT ----
        for it = 1:max_iter
            y = x_lambda + z;
            grad = A * y;
            z_new = z - mu * grad;

            % enforce finite support
            z_new(1:left-1) = 0;
            z_new(right+1:end) = 0;

            if norm(z_new - z) / max(norm(z),1e-12) < tol
                z = z_new;
                break;
            end
            z = z_new;
        end

        %% ---- SEARCH ONLY AT BOUNDARIES ----
        if N > 0
            [~,idxL] = min(abs(Zc - z(left)));
            [~,idxR] = min(abs(Zc - z(right)));

            z(left)  = Zc(idxL);
            z(right) = Zc(idxR);

            % peel
            x_lambda(left)  = x_lambda(left)  + z(left);
            x_lambda(right) = x_lambda(right) + z(right);

            z(left)  = 0;
            z(right) = 0;
        else
            % center sample
            [~,idxC] = min(abs(Zc - z(center)));
            z(center) = Zc(idxC);
            x_lambda(center) = x_lambda(center) + z(center);
        end
    end

    x_rec = x_lambda;
end


function W = DFT_matrix(N)
    % Construct DFT-like matrix used by Python code (matching indexing)
    [i,j] = meshgrid(0:N-1, 0:N-1);
    omega = exp(-2*pi*1i / N);
    W = omega .^ ((i - N/2) .* j);   % matches python formula
end

function y = high_pass_matlab(x, beta, rho, w)
    % MATLAB version of high_pass (matching python)
    % x is length-L column vector
    X_ft_hpf = fft(ifftshift(x));
    [~, idx_1] = min(abs(w - beta * rho * pi));
    [~, idx_2] = min(abs(w - (2 * pi - beta * rho * pi)));
    X_ft_hpf(1:idx_1) = 0;
    X_ft_hpf(idx_2+1:end) = 0;
    y = ifftshift(ifft(X_ft_hpf));
    y = real(y);
end

function delta_q = quant_delta(delta_rec, Lambda1,Lambda2)
    delta_q = round(delta_rec / (2*Lambda1)) * (2*Lambda2);
end
