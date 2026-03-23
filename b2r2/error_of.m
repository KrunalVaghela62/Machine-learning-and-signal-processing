function main
W=50;
E=1;
OF=2;
BW = 2 * pi * W; % Bandwidth in rad/s
% Derived parameters
Wnyq = 2 * BW; % Nyquist frequency
Tnyq = 2 * pi / Wnyq; % Nyquist sampling interval
Ws = OF * Wnyq; % Sampling frequency with oversampling
Ts = 2 * pi / Ws; % Corresponding sampling period
Td = Ts / 1000;

lambda=0.25;
N_lambda=150;
[x_orig_cont,t_d]=generate_original(E,W,OF);
[x_orig,t_s,len_ts_arr]=generate_sampled(x_orig_cont,t_d);
x_lambda = mod(real(x_orig) + lambda, 2*lambda) - lambda;


[x_rec]=pgd(x_lambda,lambda,N_lambda,Ws,BW,Ts);
%{
figure;hold;plot(t, f, 'b','LineWidth',4);
% ylim([1.1*min(f) 1.1*max(f)])
% set(gca,'YTick',[])
% set(gca, 'YtickLabel',{'0','1'})
% set(gca,'XTick',[])

% ylabel('$f(t)$','FontSize',30,'FontName','Times','Interpreter','Latex')
xlabel('$t$','FontSize',30,'FontName','Times','Interpreter','Latex')
% xlim([1 7])
% set(gca,'TickLabelInterpreter', 'Latex')
set(gca,'FontSize',30,'FontName','Times')
%}
% Plotting
figure;
plot(t_s, x_orig, '-r', 'LineWidth', 4); hold on;
plot(t_s, x_rec, 'b--', 'LineWidth', 4);

legend('Original', 'Reconstructed');
xlabel('$t$', 'FontSize', 30,'FontName','Times', 'Interpreter', 'latex');
set(gca,'FontSize',30,'FontName','Times')
end
function [x_rec]=pgd(x_lambda,lambda,N_lambda,ws,wm,Ts)
f_hat = x_lambda;
[F_rho, omega_rho, n] = construct_partial_DTFT_matrix( ws, wm, Ts);
F_rho_adj = (Ts / (2*pi)) * (F_rho)';
z = projection(N_lambda, Ts,real((F_rho_adj)*(F_rho)*(x_lambda)));
gamma_k=2/(2^10);
while N_lambda>0
for k=1:1000
error_term= (F_rho_adj)*(F_rho)*(z-f_hat);
y = z - gamma_k*(real(error_term));
z = projection(N_lambda,Ts,y);
end
z_hat = z;
z_hat = round(z_hat / (2*lambda)) * 2 * lambda;
f_hat = f_hat - z_hat;
N_lambda = N_lambda - 1;
z = projection(N_lambda,Ts,z_hat);
end
x_rec = f_hat;
end
function [x_orig_cont,t_d]=generate_original(E,W,OF)
% Parameter
NoS = 10; % Number of sinc components (set this as needed)
BW = 2 * pi * W ; % Bandwidth in rad/s
% Derived parameters
Wnyq = 2 * BW; % Nyquist frequency
Tnyq = 2 * pi / Wnyq; % Nyquist sampling interval
Ws = OF * Wnyq; % Sampling frequency with oversampling
Ts = 2 * pi / Ws; % Corresponding sampling period
Td = Ts / 1000; % Dense time axis step for plotting
CoS = 0.5 * randn(NoS, 1); % Random coefficients for sinc components
t = -1 : Td : 1; % Time axis
t = t(:); % Ensure column vector
f = zeros(length(t), 1); % Initialize signal
% Construct the bandlimited signal as a sum of sinc functions
for m = 1 : NoS
f = f + CoS(m) * sinc(BW / pi * t - (m - NoS/2)*5);
end
% Normalize signal
c = 1; %maximum value attained by signal
f = c * f / max(abs(f));
x_orig_cont=f;
t_d=t;
end
%[x_orig_cont,t_d]=generate_original(E, W,OF);
function [x_orig,t_s,len_ts_arr]=generate_sampled(x_orig_cont,t_d)
x_orig = downsample(x_orig_cont, 1000); % Sample values
t_s = downsample(t_d, 1000); % Sampling locations
len_ts_arr=length(t_s);
end
function [F_rho, omega_rho, n] = construct_partial_DTFT_matrix( ws, wm, Ts)
N=1/Ts;
% Time indices: n = -N to N
n = (-N:N); % [1 × 2N+1]

% Frequency axis: 2N+1 points between -ws/2 and ws/2
omega = linspace(-ws/2, ws/2, 2*N + 1); % [1 × 2N+1]
% Construct full DTFT matrix [2N+1 × 2N+1]
F_full = zeros(2*N+1, 2*N+1);
for k = 1:(2*N+1)
F_full(k, :) = exp(-1j * omega(k) * Ts * n);
end
% Logical index for frequencies in the partial band ρ = (-ws/2,-wm) ∪ (wm,ws/2)
idx_rho = (abs(omega) > wm);
% Keep only those rows
F_rho = F_full(idx_rho, :);
omega_rho = omega(idx_rho); % Filtered frequency values
end
%F_rho_adj = (Ts / (2*pi)) * F_rho'; % Adjoint operator ≈ inverse DTFT
function [projected_function] = projection(N_lambda, Ts, func_arr)
% Input: func_arr is (2N+1)x1 or 1x(2N+1)
N = round(1 / Ts); % Total half-width
center = N + 1; % MATLAB indexing (1-based)
idx1 = center - N_lambda;
idx2 = center + N_lambda;
% Zero out values outside [-N_lambda, N_lambda]
projected_function = func_arr;

if isrow(func_arr)
projected_function(1:idx1-1) = 0;
projected_function(idx2+1:end) = 0;
else
projected_function(1:idx1-1) = 0;
projected_function(idx2+1:end) = 0;
end
end