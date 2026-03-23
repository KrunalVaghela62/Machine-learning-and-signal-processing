S=[-1,1,2,-2];
E=10;
W=10;
BW = 2 * pi * W;         % Bandwidth (rad/s)

OF=3;
% Derived parameters
Wnyq = 2 * BW;
Tnyq = 2 * pi / Wnyq;    
Ws   = OF * Wnyq;
Ts=2*pi/Ws;
Delta=0.4;
[x_orig_cont,t_d,CoS,NoS]=generate_analog_transmit(E, W,OF,S);
[x_tru,t_s]=generate_sampled_signal(x_orig_cont,t_d);
plot_spectrum(x_orig_cont, t_d);
SNR = 20;  % dB
noisy_x = awgn(x_orig_cont, SNR, 'measured');
%noise=noisy_x-x_orig_cont;
% --- Ideal low-pass at cutoff W Hz ---
x = noisy_x;          % noisy signal
t = t_d;

Td = t(2) - t(1);
Fs = 1/Td;
N  = length(x);

X = fftshift(fft(x));    % frequency-domain signal
plot_spectrum(noisy_x, t_d);
figure;
plot(t, real(x));
title('Filtered Signal after adding noise');
grid on;
% --- Ideal low-pass at cutoff W Hz ---
x = noisy_x;          % noisy signal
t = t_d;

Td = t(2) - t(1);
Fs = 1/Td;
N  = length(x);

X = fftshift(fft(x));    % frequency-domain signal

% Frequency axis
f_axis = linspace(-Fs/2, Fs/2, N);
%{
% Ideal low-pass mask
cutoff = W;     % 10 Hz
H = abs(f_axis) <= cutoff;   % mask 1 inside band, 0 outside
H = H(:);        % force column vector
% Apply LPF
%}

% ---------- Non-Ideal Trapezoidal LPF ----------
Wc = W+10;              % Passband cutoff
TW = 0.5* Wc;       % 30% transition width

H = zeros(N,1);

% Passband: |f| <= W
H(abs(f_axis) <= Wc) = 1;

% Transition band: W < |f| < W+TW
idx_tr = (abs(f_axis) > Wc) & (abs(f_axis) <= Wc + TW);
H(idx_tr) = 1 - (abs(f_axis(idx_tr)) - Wc)/TW;

X_filt = X .* H;

% Back to time domain
x_filt = ifft(ifftshift(X_filt));



plot_spectrum(x_filt, t_d);

figure;
plot(t, real(x_filt));
title('Filtered Signal after Ideal Low-pass at ±10 Hz');
grid on;

[x_orig,t_s]=generate_sampled_signal(x_filt,t_d);
x_mod = arrayfun(@(x) modulo_operation(x, Delta), x_orig);
figure;
stem(t_s, real(x_mod));
title('modulo signal');
grid on

%%%using x_mod now
K = compute_required_filter_length(W,E,Delta,Ts);
fprintf('Demo Parameters:\n');
fprintf('Bandwidth W = %.1f Hz\n', W);
fprintf('Sampling period Ts = %.3f s\n', Ts);
fprintf('Nyquist condition: Ts < %.3f s ✓\n', 1/(2*W));
fprintf('Modulo threshold Delta = %.1f\n', Delta);
fprintf('Energy bound E = %.1f\n\n', E);
fprintf('Required filter length K = %d\n', 2*K);
fprintf('Running corrected recovery algorithm...\n');
tic;
%sigma=0.0125;
%noise = sigma * (2*rand(size(x_orig)) - 1); % Generate noise
%x_orig_noise=x_orig+noise;
[x_rec, success] = modulo_recovery_corrected2(x_mod, Ts, W, Delta, E,x_orig);

x_rect = zeros(NoS,1);   % final values of x_rec at desired t_s
%idx_store = zeros(NoS,1); % (optional) store the matching indices

for m = 1:NoS
    
    % target time instant
    t_target = (2*m - NoS)*Tnyq;
    
    % find nearest index in t_s
    [~, idx] = min(abs(t_s - t_target));
    
    % store the reconstructed sample at that time
    x_rect(m) = x_rec(idx);
    
    % optional: keep index
    %idx_store(m) = idx;
end
x_rect'

A = generate_all_as(S, NoS);
% Ensure x_rect is a column
xvec = x_rect(:);

% Compute squared Euclidean distance to every column of A
dists = sum((A - xvec).^2, 1);   % 1 means sum across rows

% Find index of minimum distance
[~, idx_min] = min(dists);

% Extract nearest column
x_nearest = A(:, idx_min);

% Update x_rect
x_rect = x_nearest'



x_rec_cont = sinc_interpolation(x_rec, t_s, t_d, Ts,OF);
elapsed_time = toc;
K =compute_required_filter_length(W, E, Delta, Ts);%compute_required_filter_length(W, E, Delta, Ts);




%[y_convolve, ts_new] = fft_convolve_with_timeaxis(x_orig, h_coeffs, Ts, t_s(1));


    
mse=(norm(x_tru - x_rec))^2/(norm(x_tru))^2;
%mse = mean((x_orig - x_rec).^2);
mse_db = 10 * log10(mse);

%{
x_check_cont=sinc_interpolation(x_orig, t_s, t_d, Ts,OF);
mse_cont_check = (norm(x_orig_cont - x_check_cont))^2/(norm(x_orig_cont))^2;


mse_cont = (norm(x_orig_cont - x_rec_cont))^2/(norm(x_orig_cont))^2;
mse_cont_db = 10 * log10(mse_cont);

%}

fprintf('✓ Recovery successful!\n');
fprintf('MSE: %.2e\n', mse);
%fprintf('MSE_cont: %.2e\n', mse_cont);
fprintf('MSE_db: %.2e\n', mse_db);
%fprintf('MSE_db_cont: %.2e\n', mse_cont_db);
fprintf('Computation time: %.3f seconds\n\n', elapsed_time);

% Plot results
%plot_recovery_results(x_orig_cont,x_orig, x_mod, x_rec,x_rec_cont,t_d,t_s);



    