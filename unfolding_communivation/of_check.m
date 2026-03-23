% MATLAB script to compare Standard QPSK vs. Combined Relay System
clear; clc; close all;

%% 1. Parameters and Weights
% Weights provided for the relay paths
w11 = 0.3; w12 = 0.6; w21 = 0.7; w22 = 0.1;

% Define a range for sigma (noise standard deviation)
% From 0.1 (low noise) to 1.2 (high noise)
sigma = linspace(0.1, 1.2, 100); 

% Q-function definition: Q(x) = 0.5 * erfc(x / sqrt(2))
q_func = @(x) 0.5 * erfc(x ./ sqrt(2));

%% 2. Equation 1: Standard QPSK (Top of Image)
% Formula: Ps1 = 2*Q(1/sigma) - Q(1/sigma)^2
Q_std = q_func(1 ./ sigma);
Ps1 = 2 * Q_std - Q_std.^2;

%% 3. Equation 2: Combined System (Bottom of Image)
% Calculating alpha and beta as defined in the system
alpha = sqrt(1 + w11^2 + w21^2) ./ sigma;
beta  = sqrt(1 + w12^2 + w22^2) ./ sigma;

% Calculating individual Q-values
Q_alpha = q_func(alpha);
Q_beta  = q_func(beta);

% Logic from your interpretation: Ps2 = Q(alpha) + Q(beta) - Q(alpha)*Q(beta)
Ps2 = Q_alpha + Q_beta - (Q_alpha .* Q_beta);

%% 4. Plotting the Results
figure('Color', 'w', 'Name', 'QPSK vs Diversity System');

% We use a log scale for the Y-axis to see the error rates clearly
semilogy(sigma, Ps1, 'b-', 'LineWidth', 2); hold on;
semilogy(sigma, Ps2, 'r--', 'LineWidth', 2);

grid on;
xlabel('\sigma (Noise Standard Deviation)');
ylabel('Symbol Error Probability (P_s)');
title('Numerical Comparison: Standard QPSK vs. Combined Relay');
legend('Standard QPSK (Eq 1)', 'Combined Relay (Eq 2)');

%% 5. Numerical Output
% Display specific values for verification at low, medium, and high noise
fprintf('Sigma\t\t Ps1 (Standard)\t Ps2 (Relay System)\n');
fprintf('---------------------------------------------------\n');
check_points = [0.2, 0.5, 0.8];
for s = check_points
    [~, idx] = min(abs(sigma - s));
    fprintf('%.2f\t\t %.2e\t\t %.2e\n', sigma(idx), Ps1(idx), Ps2(idx));
end