% run_svd_cost_pgd.m
clear; close all; clc;

%% ===================== User parameters =====================
OF = 15;
Lambda = 0.6;

E = 10;
W = 10;

S = [-1,1,2,-2];

pad = 50;   % IMPORTANT

BW = 2 * pi * W;
Wnyq = 2 * BW;
Tnyq = 2 * pi / Wnyq;

%% ================ 1) Generate signal =================
[x_orig_cont,t_d,CoS,NoS] = generate_analog_transmit(E, W, OF, S);

[x_sampled, t_s] = generate_sampled_signal(x_orig_cont, t_d, W, OF);

x_true = double(x_sampled(:));
L = length(x_true);

fprintf('Discrete length L = %d\n', L);

%% ================= Modulo =================
x_lambda = mod(x_true + Lambda, 2*Lambda) - Lambda;

%% ================= High-pass =================
rho = 1/OF;
beta = 1.02;
w = linspace(0,2*pi,L);

y = high_pass_matlab(-x_lambda, beta, rho, w);

%% ================= Build A =================
[~, idx_1] = min(abs(w - beta*rho*pi));
[~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));

diagonal = ones(L,1);
diagonal(1:idx_1+1) = 0;
diagonal(idx_2:end) = 0;

D = diag(diagonal);
F = DFT_matrix(L);
A = D * F;

%% ================= Build idx_array =================
idx_array = [];

for m = 1:NoS
    t_target = (2*m - NoS)*Tnyq;
    [~, idx2] = min(abs(t_s - t_target));

    local_idx = (idx2 - pad):(idx2 + pad);
    local_idx = local_idx(local_idx >= 1 & local_idx <= L);

    idx_array = [idx_array; local_idx(:)];
end

idx_array = unique(idx_array);
idx_array = sort(idx_array);

fprintf('Size of idx_array = %d\n', length(idx_array));

%% ================= Partition =================
all_idx = (1:L)';
idx_C = setdiff(all_idx, idx_array);

A_T = A(:, idx_array);
A_C = A(:, idx_C);

%% ================= SVD =================
[U,S,~] = svd(A_C, 'econ');

sing_vals = diag(S);
tol = 1e-10;
r = sum(sing_vals > tol);

fprintf('rank(A_C) = %d\n', r);

U2 = U(:, r+1:end);

%% ================= Initialization =================
delta_init = high_pass_matlab(-x_lambda, beta, rho, w);

z_T =delta_init(idx_array);   % start from modulo samples

%% ================= PGD =================
max_iter = 2000;
mu = 1 / norm(A_T)^2;

for iter = 1:max_iter
    
    % ===== gradient =====
    res = U2' * (A_T * z_T) - U2' * y;
    grad = 2 * A_T' * (U2 * res);
    
    % update
    z_T = z_T - mu * grad;
    
    % ===== controlled quantization =====
    delta_cont = (z_T - x_lambda(idx_array)) / (2*Lambda);
    delta_round = round(delta_cont)*2*Lambda;
    
    
    
    if mod(iter,100)==0
        fprintf('Iter %d, grad norm = %.2e\n', iter, norm(grad));
    end
end

%% ================= OUTPUT =================
diff_T = x_lambda(idx_array) - z_T;

disp('diff_T = ');
disp(diff_T.');

%% ================= Plot =================
t_selected = t_s(idx_array);

figure;
stem(t_selected, real(diff_T), 'filled');
xlabel('Time');
ylabel('x_\lambda - z_T');
title('Modulo Correction (SVD Cost + PGD)');
grid on;

%% ================= FUNCTIONS =================

function W = DFT_matrix(N)
    [i,j] = meshgrid(0:N-1, 0:N-1);
    omega = exp(-2*pi*1i / N);
    W = omega .^ ((i - N/2) .* j) / sqrt(N);
end

function y = high_pass_matlab(x, beta, rho, w)

    X = fft(ifftshift(x));

    [~, idx_1] = min(abs(w - beta*rho*pi));
    [~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));

    X(1:idx_1) = 0;
    X(idx_2:end) = 0;

    y = ifftshift(ifft(X));
    y = real(y);
end