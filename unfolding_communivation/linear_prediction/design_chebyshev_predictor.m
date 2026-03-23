function h_coeffs = design_chebyshev_predictor(K, W, Ts)
% DESIGN_CHEBYSHEV_PREDICTOR Designs predictor using Chebyshev polynomials
% Implements equation (7) from the paper
    
    % Compute interval bounds for Chebyshev polynomial
    a = 2 * cos(2*pi*W*Ts);  % Lower bound
    b = 2;                   % Upper bound
    
    % Generate Chebyshev polynomial T^[a,b]_K
    cheb_poly = generate_chebyshev_polynomial(K, a, b);
    
    % Construct p_K(z) = z^K * T^[a,b]_K(z + z^(-1))
    % This gives us the monic polynomial as in equation (9)
    p_coeffs = construct_predictor_polynomial(cheb_poly, K);
    
    % Extract filter coefficients h_1, ..., h_{2K}
    % From p_K(z) = -h_{2K}z^{2K} - ... - h_1z + 1
    h_coeffs = -p_coeffs(1:end-1);  % Remove constant term
end

function T_poly = generate_chebyshev_polynomial(K, a, b)
% GENERATE_CHEBYSHEV_POLYNOMIAL Creates Chebyshev polynomial on interval [a,b]
    
    if K == 0
        T_poly = 1;
        return;
    end
    
    % Transform to standard interval [-1, 1]
    % y = 2/(b-a) * (x - (a+b)/2)
    scale_factor = 2 / (b - a);
    offset = (a + b) / 2;
    
    % Generate Chebyshev polynomial using recurrence relation
    T_prev_prev = 1;                    % T_0
    T_prev = [scale_factor, -scale_factor * offset];  % T_1 transformed
    
    if K == 1
        T_poly = T_prev;
        return;
    end
    
    % Recurrence: T_{k+1}(y) = 2y*T_k(y) - T_{k-1}(y)
    for k = 2:K
        % 2y * T_k(y)
        term1 = conv([2*scale_factor, -2*scale_factor*offset], T_prev);
        
        % Pad T_{k-1} to match dimensions
        if length(T_prev_prev) < length(term1)
            T_prev_prev = [zeros(1, length(term1) - length(T_prev_prev)), T_prev_prev];
        elseif length(term1) < length(T_prev_prev)
            term1 = [zeros(1, length(T_prev_prev) - length(term1)), term1];
        end
        
        % T_{k+1} = 2y*T_k - T_{k-1}
        T_current = term1 - T_prev_prev;
        T_prev_prev = T_prev;
        T_prev = T_current;
    end
    
    % Apply scaling factor from equation (6)
    scaling = 2 * ((b - a) / 4)^K;
    T_poly = scaling * T_prev;
end

function p_coeffs = construct_predictor_polynomial(cheb_coeffs, K)
% Builds p_K(z) = z^K * T_K(z + z^{-1})
% cheb_coeffs: coefficients of T_K(x), length K+1, ordered from highest degree to constant term
% The function reverses them internally to constant-first order

    % Reverse input coefficients to ascending order (constant term first)
    cheb_coeffs = fliplr(cheb_coeffs);

    % Initialize polynomial coefficients vector for degree 2K polynomial
    p_coeffs = zeros(1, 2*K + 1);
    
    for i = 0:K
        c = cheb_coeffs(i+1);  % MATLAB 1-based indexing
        if c == 0
            continue;
        end
        
        % Expand (z + z^{-1})^i
        term = expand_z_plus_z_inv_power(i);  % length 2i+1
        
        % Calculate shift amount: shift right by (K - i)
        shift = K - i;
        
        % Add c * term shifted by 'shift' to p_coeffs
        p_coeffs(shift + (1:length(term))) = p_coeffs(shift + (1:length(term))) + c * term;
    end
    
    % Flip coefficients to MATLAB polynomial order (highest degree first)
    p_coeffs = fliplr(p_coeffs);
end



function expanded = expand_z_plus_z_inv_power(n)
% Expands (z + z^{-1})^n as a vector of coefficients
% Length = 2n + 1, center at n+1 corresponds to z^0

    expanded = zeros(1, 2*n + 1);
    center = n + 1;  % index for z^0

    if n == 0
        expanded(center) = 1;
        return;
    end

    for k = 0:n
        coeff = nchoosek(n, k);
        power = n - 2*k;  % power of z
        idx = center + power;
        % Accumulate coefficients (safe practice)
        expanded(idx) = expanded(idx) + coeff;
    end
end