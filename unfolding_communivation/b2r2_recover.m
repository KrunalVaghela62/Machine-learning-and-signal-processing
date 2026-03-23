function z_hat = b2r2_recover(x_mod, W, Ts, Delta, Nlambda_init, iters)
% ---------------------------------------------------------
% x_mod        : modulo samples (Ns x 1)
% W            : signal bandwidth (Hz)
% Ts           : sampling interval
% Delta        : modulo width
% Nlambda_init : initial half-support for PGD
% iters        : PGD iterations per stage
% ---------------------------------------------------------

    Ns = length(x_mod);

    % ============================================================
    % 1) Build frequency mask ρ for high-pass operation (corrected)
    % ============================================================
    omega_s = 2*pi/Ts;
    df = omega_s / Ns;

    % Correct FFT-aligned frequency grid
    omega = fftshift((-floor(Ns/2):ceil(Ns/2)-1)' * df);

    omega_m = 2*pi*W;
    rho_mask = (abs(omega) > omega_m);     % 1 = out-of-band

    % ============================================================
    % 2) Improved INITIALIZATION
    % ============================================================
    X = fft(x_mod);
    X_shift = fftshift(X);
    X_rho = X_shift .* rho_mask;
    z = real(ifft(ifftshift(X_rho)));       % correct shift order

    % Scale up initialization (important!)
    %z = z * 5;       % Helps escape zero solution

    % ============================================================
    % 3) Sequential SUPPORT REDUCTION
    % ============================================================
    Nlambda = Nlambda_init;
    center = floor(Ns/2) + 1;

    while Nlambda >= 1
        fprintf('--- PGD stage with Nλ = %d ---\n', Nlambda);

        % Support mask
        supp = center + (-Nlambda:Nlambda);
        mask_time = false(Ns,1);
        valid_idx = supp(supp >= 1 & supp <= Ns);
        mask_time(valid_idx) = true;

        % ---------------- PGD Iterations ----------------
        for k = 1:iters

            % Compute gradient
            d = z - x_mod;
            D = fft(d);
            D_shift = fftshift(D);
            D_hp = D_shift .* rho_mask;
            grad = real(ifft(ifftshift(D_hp)));

            % Gradient descent
            gamma = 0.1;              % larger step improves convergence
            y = z - gamma * grad;

            % Projection
            y(~mask_time) = 0;

            % Quantization onto Δℤ
            z = Delta * round(y / Delta);

        end

        % Shrink support for next stage
        Nlambda = Nlambda - 1;
    end

    z_hat = z;
end
