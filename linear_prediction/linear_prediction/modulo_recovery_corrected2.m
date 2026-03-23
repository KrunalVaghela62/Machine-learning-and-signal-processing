function [x_recovered, success] = modulo_recovery_corrected2(x_star, Ts, W, Delta, E,x_orig,epsilon)
% MODULO_RECOVERY_CORRECTED Recovers bandlimited signal from modulo samples
% Based on Romanov-Ordentlich paper implementation
%
% Inputs:
%   x_star - modulo-reduced samples
%   Ts     - sampling period (must be < 1/(2*W))
%   W      - signal bandwidth
%   Delta  - modulo threshold
%   E      - signal energy bound
%
% Outputs:
%   x_recovered - recovered original samples
%   success     - recovery success flag
addpath('C:\Users\Krunal\OneDrive\Documents\MATLAB\new_algo');

    % Verify Nyquist condition
    if Ts > 1/(2*W)
        error('Sampling rate must exceed Nyquist rate: Ts < 1/(2*W)');
    end
    
    N = length(x_star);
    x_recovered = zeros(size(x_star));
    success = true;
    
    
    % Design Chebyshev-based prediction filter
    K =compute_required_filter_length(W, E, 2*epsilon, Ts);%compute_required_filter_length(W, E, Delta, Ts);
    filter_length = 2*K;  % As per equation (9) in the paper
    h_coeffs = design_chebyshev_predictor(K, W, Ts);
    

    % Find initialization point N where |x_n| < Delta/2
    % Use conservative estimate - start from beginning for simplicity
    N_init = estimate_N_lam(x_star,h_coeffs,Delta);

    
    % Initialize with unfolded samples (assuming they're small enough)
    for n = 1:N_init
        x_recovered(n) = x_star(n);
    end

    % Sequential recovery using the prediction-correction scheme
    for n = N_init+1 :N
        % Predict current sample from past samples
        x_pred = compute_prediction(x_recovered, n, h_coeffs, filter_length);
        
        % Compute modulo-reduced prediction error
        e_star = modulo_operation(x_star(n) - x_pred, Delta);
        
        % Check if prediction error is within bounds
        if abs(e_star) >= Delta/2
            warning('Recovery failed at sample %d: prediction error too large', n);
            success = false;
            break;
        end
        
        % Recover current sample: x_n = x_pred + e_star
        x_recovered(n) = x_pred + e_star;
    end
end
