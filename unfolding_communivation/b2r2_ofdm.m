clc; clear; close all;

%% ================= PARAMETERS =================
W  = 10;                 % Bandwidth (Hz)
OF = 4;                  % Oversampling factor
S  = [-1 -2 1 2];        % Symbol alphabet
M  = 5;                  % Number of OFDM symbols

lambda = 2;
Delta  = 2*lambda;

%% ================= GENERATE OFDM =================
[x_cont, t_d, CoS_true, NoS] = generate_ofdm_signal(W, OF, S, M);

disp('True symbols:');
disp(CoS_true.');

%% ================= SAMPLE SIGNAL =================
[x_s, t_s] = generate_sampled_signal(x_cont, t_d,W,OF);
Td = t_s(2) - t_s(1);
L  = length(x_s);

x_true = x_s(:);     % COMPLEX samples

%% ================= MODULO ADC (I/Q SEPARATELY) =================
x_lambda_real = mod(real(x_true) + lambda, 2*lambda) - lambda;
x_lambda_imag = mod(imag(x_true) + lambda, 2*lambda) - lambda;

%% ================= B2R2 RECONSTRUCTION =================
% -------- REAL PART --------
X_ft_lambda_real = fft(ifftshift(x_lambda_real));
delta_true_real  = real(x_true) - x_lambda_real;

M_est_real = M_lambda(delta_true_real, lambda, L);

x_rec_real = BBRR_matlab_real_signal( ...
    x_lambda_real, X_ft_lambda_real, M_est_real, OF, lambda, L);

% -------- IMAG PART --------
X_ft_lambda_imag = fft(ifftshift(x_lambda_imag));
delta_true_imag  = imag(x_true) - x_lambda_imag;

M_est_imag = M_lambda(delta_true_imag, lambda, L);

x_rec_imag = BBRR_matlab_real_signal( ...
    x_lambda_imag, X_ft_lambda_imag, M_est_imag, OF, lambda, L);

% -------- COMBINE --------
x_rec = x_rec_real + 1j*x_rec_imag;
%% ================= OFDM SYMBOL DETECTION =================
%{
X_rec = fftshift(fft(x_rec)) * Td;
f = (-L/2:L/2-1) / (L*Td);

X_true = fftshift(fft(x_true)) * Td;

Delta_f = W / NoS;
a_hat = zeros(NoS,1);

for k = 1:NoS
    mk = 2*k - (NoS + 1);      % SAME indexing as transmitter
    fk = Delta_f * mk;

    [~, idx] = min(abs(f - fk));
    a_hat(k) = X_rec(idx);
end


%% ================= SYMBOL SLICING =================
est_CoS = zeros(NoS,1);

for k = 1:NoS
    [~, id] = min(abs(abs(a_hat(k)) - abs(S)));

    est_CoS(k) = S(id);
end

%% ================= RESULTS =================
disp('Estimated symbols (B2R2):');
disp(est_CoS.');

fprintf('Symbol error norm = %.3e\n', norm(est_CoS - CoS_true));
%}
%% ================= PLOT ORIGINAL vs RECOVERED =================

n = (0:L-1).';                 % sample index
center = floor(L/2) + 1;
win = round(0.1*L);            % plot 10% window around center

idx = (center-win):(center+win);
idx(idx < 1 | idx > L) = [];

P= build_H_matrix(W, NoS,t_s);
P=(P*dftmtx(NoS)')/sqrt(NoS);

est_CoS= round(real(inv(P'*P)*P'*x_rec))
figure('Color','w','Position',[200 200 1000 450]);

%% ---- REAL PART ----
subplot(1,2,1);
plot(n(idx), real(x_true(idx)), '-r','LineWidth',1.4); hold on;
plot(n(idx), real(x_rec(idx)), '--b','LineWidth',1.4);
grid on;
xlabel('Sample index');
ylabel('Amplitude');
title('Real Part: Original vs B2R2 Recovered');
legend('Original','Recovered');

%% ---- IMAG PART ----
subplot(1,2,2);
plot(n(idx), imag(x_true(idx)), '-r','LineWidth',1.4); hold on;
plot(n(idx), imag(x_rec(idx)), '--b','LineWidth',1.4);
grid on;
xlabel('Sample index');
ylabel('Amplitude');
title('Imaginary Part: Original vs B2R2 Recovered');
legend('Original','Recovered');





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

function x_rec = BBRR_matlab_real_signal(x_lambda, X_ft_lambda, M, OF, Lambda, L)
    % MATLAB translation of the Python reconstruction_method.BBRR
    % This version uses rho = 1/(2*OF) to match your sampling.
    x_lambda = x_lambda(:);
    rho = 1 / (OF);    % <<< IMPORTANT: note factor 2 here for your sampled analog signal
    N = floor(M/2);
    beta = 1.05;
    decay = 0.999;
    reg = 1e-6;
    momentum = 0.9;
    epsilon = 1e-3 * (Lambda^1.5 * OF);
    w = linspace(0, 2*pi, L);   % normalized frequency vector like Python

    % Build the frequency mask (diagonal) for high-pass band
    [~, idx_1] = min(abs(w - beta*rho*pi));
    [~, idx_2] = min(abs(w - (2*pi - beta*rho*pi)));
    diagonal = ones(L,1);
    diagonal(1:idx_1+1) = 0;
    diagonal(idx_2:end) = 0;
    D = diag(diagonal);
    F = DFT_matrix(L);   % potentially large; ok for L ~ 1024
    BB = D * F;
    dev_matrix = (BB' * BB);

    if OF == 2
        step_size = 2 / L;
    else
        step_size = 2 / (idx_2 - idx_1);
    end

    % initialization
    mu = step_size;
    mom = momentum;
    change = zeros(L,1);
    d1 = 0; d2 = 0;

    % high_pass initialization (MATLAB implementation)
    delta_rec = high_pass_matlab(-x_lambda, beta, rho, w);
    center = floor(L/2) + 1;
    % projection within initial support (center +/- N)
    leftCut = max(1, center - N);
    rightCut = min(L, center + N);
    delta_rec(1 : leftCut-1) = 0;
    delta_rec(rightCut+1 : end) = 0;
    delta_rec = double(delta_rec);

    max_iter = 5e5;
    for iter = 1:max_iter
        % PGD step
        vector = x_lambda + delta_rec;
        grad = dev_matrix * vector + reg * delta_rec;
        change = mom * change + mu * real(grad);
        delta_rec = delta_rec - change;

        % Projection: zero outside support (center +- N)
        delta_rec(1 : leftCut-1) = 0;
        delta_rec(rightCut+1 : end) = 0;

        % decay params
        mu = decay * mu;
        mom = decay * mom;

        if mod(iter,5) == 0
            % Compare boundary changes
            idxL = leftCut;
            idxR = rightCut;
            d1_new = delta_rec(idxL);
            d2_new = delta_rec(idxR);
            if max(abs(d1_new - d1), abs(d2_new - d2)) < epsilon
                % Quantize delta
                delta_rec = quant_delta(delta_rec, Lambda);
                if N < 1
                    x_lambda(center) = x_lambda(center) + delta_rec(center);
                    break;
                else
                    mu = step_size;
                    mom = momentum;
                    x_lambda(idxL) = x_lambda(idxL) + delta_rec(idxL);
                    x_lambda(idxR) = x_lambda(idxR) + delta_rec(idxR);
                    % shrink support
                    N = N - 1;
                    leftCut = max(1, center - N);
                    rightCut = min(L, center + N);
                end
                % update d1,d2 for next stage
                if N >= 0
                    d1 = delta_rec(leftCut);
                    d2 = delta_rec(rightCut);
                else
                    d1 = 0; d2 = 0;
                end
            else
                d1 = d1_new; d2 = d2_new;
            end
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

function delta_q = quant_delta(delta_rec, Lambda)
    delta_q = round(delta_rec / (2*Lambda)) * (2*Lambda);
end
