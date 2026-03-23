clc; clear; close all;

%% ================= PARAMETERS =================
E  = 0.01;
W  = 20;
OF = 8;
S  = [-2 -1 1 2];

Lambda = 0.2;
Delta  = 2*Lambda;

%% ================= SIGNAL GENERATION =================
[x_orig_cont, t_d, CoS, NoS] = generate_analog_transmit(E, W, OF, S);
[x_orig, t_s] = generate_sampled_signal(x_orig_cont, t_d,W,OF);

x_orig = x_orig(:);
L = length(x_orig);
center = floor(L/2) + 1;

%% ================= MODULO FOLDING =================
x_lambda = mod(x_orig + Lambda, Delta) - Lambda;
z_true   = x_orig - x_lambda;

%% ================= SEARCH SET (COMPLETE) =================
s_max = max(abs(S));
c1 = 20;
[c_n, ~] = possible_z_values(c1, Delta);
Zc = c_n * Delta;

fprintf('Signal length L = %d\n', L);
fprintf('Search set size |Z| = %d\n', length(Zc));

%% ================= ESTIMATE SUPPORT SIZE =================
M = M_lambda(z_true, Lambda, L);
N0 = floor(M/2);
fprintf('Estimated support half-width N = %d\n', N0);

%% ================= SEARCH-BASED B2R2 =================
x_rec = B2R2_search_support_peeling(x_lambda, OF, Lambda, Zc, N0);

%% ================= METRICS =================
NMSE = norm(x_orig - x_rec)^2 / norm(x_orig)^2;
fprintf('Final NMSE = %.3e\n', NMSE);

%% ================= PLOTS =================
figure;
plot(t_s, x_orig,'k','LineWidth',1.5); hold on;
plot(t_s, x_lambda,'r');
legend('Original','Folded');
title('Original vs Folded');

figure;
plot(t_s, z_true,'b');
title('True Residual');

figure;
plot(t_s, x_orig,'k','LineWidth',1.5); hold on;
plot(t_s, x_rec,'r');
legend('Original','Recovered');
title('Search-based B2R^2 with Support Peeling');
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
    mu = 2/L;
    max_iter = 3000;
    tol = 1e-6;

    %% ===== Initialization =====
    z = high_pass_matlab(-x_lambda, beta, rho, w);

    %% ===== SUPPORT PEELING LOOP =====
    for N = N0:-1:0

        left  = center - N;
        right = center + N;

        %% PGD WITH FIXED SUPPORT
        for it = 1:max_iter
            y = x_lambda + z;
            grad = A * y;
            z_new = z - mu * grad;

            % enforce finite support
            z_new(1:left-1) = 0;
            z_new(right+1:end) = 0;

            if norm(z_new - z)/max(norm(z),1e-12) < tol
                break;
            end
            z = z_new;
        end

        %% SEARCH ONLY AT BOUNDARY INDICES
        if N > 0
            [~,idxL] = min(abs(Zc - z(left)));
            [~,idxR] = min(abs(Zc - z(right)));

            z(left)  = Zc(idxL);
            z(right) = Zc(idxR);

            % update folded signal (peeling)
            x_lambda(left)  = x_lambda(left)  + z(left);
            x_lambda(right) = x_lambda(right) + z(right);

            % remove peeled values
            z(left) = 0;
            z(right)= 0;
        else
            % center index
            [~,idxC] = min(abs(Zc - z(center)));
            z(center) = Zc(idxC);
            x_lambda(center) = x_lambda(center) + z(center);
        end
    end

    x_rec = x_lambda;
end
function M = M_lambda(delta, Lambda, L)
    thr = 0.1 * Lambda;
    idx = find(abs(delta) > thr);
    if isempty(idx), M = 0; return; end
    m_min = idx(1);
    m_max = idx(end);
    M = floor(2 * max(m_max - 0.5*L, 0.5*L - m_min) + 1);
end

function W = DFT_matrix(N)
    [i,j] = meshgrid(0:N-1,0:N-1);
    omega = exp(-2*pi*1i/N);
    W = omega.^((i - N/2).*j);
end

function y = high_pass_matlab(x, beta, rho, w)
    X = fft(ifftshift(x));
    [~,i1] = min(abs(w - beta*rho*pi));
    [~,i2] = min(abs(w - (2*pi - beta*rho*pi)));
    X(1:i1) = 0;
    X(i2:end) = 0;
    y = real(ifftshift(ifft(X)));
end
function [c_n,n]=possible_z_values(c,delta)%z=delta*c_n
    n = floor((c/delta)+0.5);
    c_n=-n:n;
end


