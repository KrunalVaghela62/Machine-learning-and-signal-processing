clc; clear; close all;

% ----- sigma range -----
sigma_vec = linspace(0.05,1.0,40);

Pe_nyq  = zeros(size(sigma_vec));
Pe_over = zeros(size(sigma_vec));

% ----- CASE 1: Nyquist sinc (2 samples) -----
w11 = 0; w12 = 0; w21 = 0; w22 = 0;

for k = 1:length(sigma_vec)
    sigma = sigma_vec(k);
    Pe_nyq(k) = compute_Pe_from_w(w11,w12,w21,w22,sigma);
end

% ----- CASE 2: oversampled (nonzero positive w's < 1) -----
w11 = 0.4;  w12 = 0.7;
w21 = 0.8;  w22 = 0.2;





for k = 1:length(sigma_vec)
    sigma = sigma_vec(k);
    Pe_over(k) = compute_Pe_from_w(w11,w12,w21,w22,sigma);
end

% ----- plot -----
figure('Color','w'); hold on; grid on;
plot(sigma_vec, Pe_nyq , 'o-', 'LineWidth', 2);
plot(sigma_vec, Pe_over, 's--', 'LineWidth', 2);

xlabel('\sigma (noise standard deviation)');
ylabel('P_e (symbol-combination error)');
title('ML Error Probability: Nyquist vs Oversampled');
legend('Nyquist sinc (2 samples)', 'Nonzero w_{ij}', 'Location','best');

set(gca,'YScale','log');
function Pe = compute_Pe_from_w(w11,w12,w21,w22,sigma)
% Computes symbol-combination error probability using
% rigorous ML + bivariate Gaussian formulation

% ----- constants -----
C1 = 1 + w11^2 + w21^2;
C2 = 1 + w12^2 + w22^2;

% ----- covariance matrix of (Z1,Z2) -----
Sigma = sigma^2 * ...
    [ 1 + w11^2 + w21^2 ,  w11*w12 + w21*w22 ;
      w11*w12 + w21*w22 ,  1 + w12^2 + w22^2 ];

mu = [0 0];

% ----- integration limits -----
lower = [-C1 , -C2];
upper = [ inf ,  inf ];

% ----- probability of correct detection -----
Pc = mvncdf(lower, upper, mu, Sigma);

% ----- probability of error -----
Pe = 1 - Pc;
end
