clc; clear; close all;

%% ================= PARAMETERS =================
W  = 10;
OF = 4;
S  = [1,-1,2,-2]
M  = 5;

lambda = 0.4;
Delta  = 2*lambda;
Eb=1/(M*log2(length(S)));
%% ================= GENERATE OFDM =================
[x_ofdm_cont, t_d, CoS, NoS] = generate_ofdm_signal(W, OF, S, M);
%[x_ofdm_cont, t_d, CoS, NoS] = generate_analog_transmit(E, W, OF, S);

disp('True symbols:');
disp(CoS.');

%% ================= SAMPLE SIGNAL =================
[x_s, t_s] = generate_sampled_signal(x_ofdm_cont, t_d,W,OF);
%sigma_x=sqrt(mean(abs(x_s).^2));
sigma = 0.4;
noise = sigma * randn(length(t_d), 1);

% Sampling parameters
Td = t_d(2) - t_d(1);
Fs = 1 / Td;
x_s=x_s;
lpf_noise = lowpass(noise, W, Fs);
lpf_noise=generate_sampled_signal(lpf_noise,t_d,W, OF);

% Add bandlimited noise
x_s = x_s + lpf_noise;
figure;
plot(t_s, real(x_s), 'b', ...
         t_s, imag(x_s), 'r');
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('OFDM Signal (Time Domain)');
    legend('Real','Imag');
    grid on;
bits_send=qam2bits(CoS,4);

%% ================= MODULO ADC (COMPLEX) =================
x_mod = arrayfun(@(v) modulo_operation(real(v), Delta), x_s) ...
      + 1j*arrayfun(@(v) modulo_operation(imag(v), Delta), x_s);

%% ================= BUILD MATCHED P-MATRIX =================
%Pmat = build_P_ofdm_matrix(t_s, W, NoS);
Pmat=build_H_matrix(W, NoS, t_s);
Pmat=Pmat*dftmtx(NoS)';
Pmat=Pmat/sqrt(NoS);
%% ================= ML DICTIONARY =================
A = generate_all_as(S, NoS);   % NoS x |S|^NoS
x_all = Pmat * A;              % COMPLEX hypotheses

%% ================= EXHAUSTIVE ML =================
best_idx  = 1;
best_dist = inf;

for k = 1:size(x_all,2)

    cand = x_all(:,k);

    cand=cand;
    cand_mod = arrayfun(@(v) modulo_operation(real(v), Delta), cand) ...
             + 1j*arrayfun(@(v) modulo_operation(imag(v), Delta), cand);
    
    d = (norm(x_mod - cand_mod))^2;

    if d < best_dist
        best_dist = d;
        best_idx  = k;
    end
end

est_CoS = A(:,best_idx);
est_CoS=est_CoS';
bits_rec=qam2bits(est_CoS,4)

%% ================= RESULTS =================
disp('Estimated symbols:');
disp(est_CoS.');


%{
clc; clear; close all;

%% ================= PARAMETERS =================
W  = 10;
OF = 4;
S  = [-2 -1 1 2];
M  = 5;

%% ================= GENERATE SIGNAL =================
[x_ofdm_cont, t_d, CoS, NoS] = generate_ofdm_signal(W, OF, S, M);

Td = t_d(2) - t_d(1);
N  = length(t_d);

%% ================= NUMERICAL FOURIER TRANSFORM =================
X_fft = fftshift(fft(x_ofdm_cont)) * Td;

f = (-N/2:N/2-1) / (N*Td);

%% ================= THEORETICAL FOURIER TRANSFORM =================
Delta_f = W / NoS;
T       = NoS / W;

X_theory = zeros(size(f));

for k = 1:NoS
    mk = 2*k - (NoS + 1);
    fk = Delta_f * mk;

    % sinc(x) in MATLAB = sin(pi x)/(pi x)
    X_theory = X_theory + ...
        CoS(k) * sinc( T * (f - fk) );
end


%% ================= PLOTS =================
figure;

subplot(3,1,1);
plot(f, abs(X_fft), 'k', 'LineWidth', 1.5);
title('Numerical Fourier Transform |X_{FFT}(f)|');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
xlim([-W W]);

subplot(3,1,2);
plot(f, abs(X_theory), 'r--', 'LineWidth', 1.5);
title('Theoretical Fourier Transform |X_{theory}(f)|');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
xlim([-W W]);

subplot(3,1,3);
plot(f, abs(X_fft - X_theory), 'b', 'LineWidth', 1.2);
title(sprintf('Absolute Error |X_{FFT} - X_{theory}|  (MSE = %.2e)', mse_val));
xlabel('Frequency (Hz)');
ylabel('Error Magnitude');
grid on;
xlim([-W W]);
%}