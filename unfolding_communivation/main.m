%% ================= PARAMETERS =================
S  = [-1,1,2,-2];
E  = 10;
W  = 10;          % cutoff in Hz
OF = 4;
lambda=0.2;
Delta=2*lambda;
[x_orig_cont, t_d, CoS, NoS] = generate_analog_transmit(E, W, OF, S);
no_noise=x_orig_cont;
figure;
plot(t_d,x_orig_cont,'LineWidth',1.5)
title('no noise')
xlabel('T')
ylabel('|X(T)|')
grid on
%% ========== NICE MODULO PLOT (LIKE IMAGE) ==========

figure; hold on;

% Original continuous signal
plot(t_d, x_orig_cont, 'b', 'LineWidth', 1.8);
x_mod=modulo_operation(x_orig_cont, Delta);
% Modulo version
plot(t_d, x_mod, 'r--', 'LineWidth', 1.6);

yl1 = yline(lambda,  'k', 'LineWidth',1.2);
yl2 = yline(-lambda, 'k', 'LineWidth',1.2);

yl1.Label = '$\lambda$';
yl2.Label = '$-\lambda$';

yl1.Interpreter = 'latex';
yl2.Interpreter = 'latex';

yl1.LabelVerticalAlignment = 'bottom';
yl2.LabelVerticalAlignment = 'top';
% Optional light shading for ADC range
patch([t_d(1) t_d(end) t_d(end) t_d(1)], ...
      [lambda lambda -lambda -lambda], ...
      [0.85 0.9 1], 'FaceAlpha',0.15, 'EdgeColor','none');

uistack(findobj(gca,'Type','line'),'top');  % keep signals on top

xlabel('Time');
ylabel('Amplitude');
title('Original vs Modulo Signal');
legend('Original','Modulo','Location','best');

grid on;
box on;
xlim([min(t_d) max(t_d)]);
Td = t_d(2) - t_d(1);
Fs = 1/Td;
N  = length(t_d);

disp("Sampling Frequency Fs = ")
disp(Fs)

%% ================= WHITE NOISE =================
sigma = 0.2*3;
noise = sigma * randn(N,1);
x=x_orig_cont;
x_orig_cont=x_orig_cont+noise;
figure;
plot(t_d,x_orig_cont,'LineWidth',1.5)
title('no lowpass noise')
xlabel('T')
ylabel('|X(T)|')
grid on
x_noisy=x_orig_cont;
%% ================= LOWPASS =================
%lpf_noise = lowpass(noise, W, Fs);
h = 2*W * sinc(2*W*t_d);   % MATLAB sinc = sin(pi x)/(pi x)
%% ================= PSD (THIS IS IMPORTANT) =================
%y = conv(x_orig_cont, h, 'same')*Td;
y = lowpass(x_noisy,W,Fs,ImpulseResponse="iir",Steepness=0.95);

figure;
plot(t_d,y,'LineWidth',1.5)
title('bandpass noise filtered signal')
xlabel('T')
ylabel('|X(T)|')
grid on
figure;
plot(t_d,no_noise-y,'LineWidth',1.5)
title('bandpass noise')
xlabel('T')
ylabel('|X(T)|')
grid on
[x_orig,t_s]=generate_sampled_signal(y,t_d,W,OF);
%% FFTs
X      = fftshift(fft(x))*Td;
Xn     = fftshift(fft(x_noisy))*Td;
Y      = fftshift(fft(y))*Td;
f = (-N/2:N/2-1)*(Fs/N);
%% Plot spectra
figure;

subplot(3,1,1)
plot(f, abs(X),'LineWidth',1.5)
title('Spectrum of Original Signal')
xlabel('Frequency (Hz)')
ylabel('|X(f)|')
xlim([-500 500])
grid on

subplot(3,1,2)
plot(f, abs(Xn),'LineWidth',1.5)
title('Spectrum of Signal + Noise')
xlabel('Frequency (Hz)')
ylabel('|X_n(f)|')
xlim([-500 500])
grid on

subplot(3,1,3)
plot(f,abs(Y),'LineWidth',1.5)
title('Spectrum After Ideal Lowpass')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')
xlim([-500 500])
grid on
%{
% ============================
% NON-IDEAL LOW-PASS FILTER
% (trapezoid with 30% transition)
% ============================

x = noisy_x;          
t = t_d;

Td = t(2) - t(1);
Fs = 1/Td;
N  = length(x);

X = fftshift(fft(x));    
plot_spectrum(noisy_x, t_d);

figure;
plot(t, real(x));
title('Signal + Noise');
grid on;

% Frequency axis
f_axis = linspace(-Fs/2, Fs/2, N).';

% ---------- Non-Ideal Trapezoidal LPF ----------
Wc = W;              % Passband cutoff
TW = 0.3 * Wc;       % 30% transition width

H = zeros(N,1);

% Passband: |f| <= W
H(abs(f_axis) <= Wc) = 1;

% Transition band: W < |f| < W+TW
idx_tr = (abs(f_axis) > Wc) & (abs(f_axis) <= Wc + TW);
H(idx_tr) = 1 - (abs(f_axis(idx_tr)) - Wc)/TW;

% Stopband: H=0 automatically
% ----------------------------------------------

% Apply LPF
X_filt = X .* H;
x_filt = ifft(ifftshift(X_filt));

plot_spectrum(x_filt, t_d);

figure;
plot(t, real(x_filt));
title('Filtered Signal (Non-Ideal LPF)');
grid on;
%}


% ========== Sampling =============

% ========== Modulo ===============
x_mod = arrayfun(@(x) modulo_operation(x, Delta), x_orig);


% ========== Exhaustive ML ==========
A = generate_all_as(S, NoS);

Hmat = build_H_matrix(W, NoS, t_s);
x_orig_all = Hmat * A;

[Ns, NumComb] = size(x_orig_all);

x_mod_all = zeros(Ns, NumComb);

for k = 1:NumComb
    x_col = x_orig_all(:, k);
    %x_col=x_col/sqrt(mean(x_col.^2));
    x_mod_all(:, k) = arrayfun(@(x) modulo_operation(x, Delta), x_col);
end

best_idx = 1;
best_dist = inf;

for k = 1:NumComb
    diff_vec = x_mod - x_mod_all(:, k);
    dist_k = (norm(diff_vec))^2;
    if dist_k < best_dist
        best_dist = dist_k;
        best_idx = k;
    end
end

est_CoS = A(:,best_idx)';
