function main()
% DEMO_CORRECTED_RECOVERY with GPU acceleration
% Demonstrates modulo recovery with sinc interpolation and noise

    % =========================
    % Parameters
    % =========================
    E       = 50;          % Target signal energy
    lambda  = 0.1;       % Scaling parameter
    Delta   = 2 * lambda;  % Modulo threshold
    W       = 50;          % Bandwidth (Hz)
    OF      = 2;          % Oversampling factor
    BW      = 2 * pi * W;  % Bandwidth in rad/s

    % Sampling parameters
    Wnyq = 2 * BW;          % Nyquist frequency
    Ws   = OF * Wnyq;       % Sampling frequency
    Ts   = 2 * pi / Ws;     % Sampling period

    % Compute required filter length
    K = compute_required_filter_length(W, E, Delta, Ts);

    % =========================
    % Display parameters
    % =========================
    fprintf('Demo Parameters:\n');
    fprintf('Bandwidth W         = %.1f Hz\n', W);
    fprintf('Sampling period Ts  = %.3f s\n', Ts);
    fprintf('Nyquist condition   : Ts < %.3f s ✓\n', 1 / (2 * W));
    fprintf('Modulo threshold Δ  = %.1f\n', Delta);
    fprintf('Energy bound E      = %.1f\n', E);
    fprintf('Required filter K   = %d\n\n', 2 * K);

    % =========================
    % Generate signals (GPU)
    % =========================
    [x_orig_cont, t_d] = generate_original_signal(E, W, OF);
    [x_orig_cpu, t_s]  = generate_sampled_signal(x_orig_cont, t_d);

    % Move signals to GPU
    x_orig = gpuArray(x_orig_cpu);
    t_s    = gpuArray(t_s);
    t_d    = gpuArray(t_d);

    % =========================
    % Add bounded uniform noise
    % =========================
    

    % =========================
    % Modulo operation
    % =========================
    x_mod = modulo_operation(x_orig, Delta);
    snr=20;

    % Add noise (on GPU)
    noise_bound = lambda*10^(-snr/20);
    noise = (2*rand(size(x_mod),'gpuArray') - 1) * noise_bound;
    x_mod = x_mod + noise;
    % =========================
    % Recovery
    % =========================
    fprintf('Running corrected recovery algorithm...\n');
    tic;
    [x_rec, success] = modulo_recovery_corrected(x_mod, Ts, W, Delta, E, x_orig);
    x_rec_cont = sinc_interpolation(x_rec, t_s, t_d, Ts, OF);
    elapsed_time = toc;

    % =========================
    % Error metrics
    % =========================
    mse     = (norm(x_orig - x_rec))^2 / (norm(x_orig))^2;
    mse_db  = gather(10 * log10(mse));

    x_check_cont    = sinc_interpolation(x_orig, t_s, t_d, Ts, OF);
    mse_cont_check  = (norm(x_orig_cont - x_check_cont))^2 / (norm(x_orig_cont))^2;

    mse_cont     = (norm(x_orig_cont - x_rec_cont))^2 / (norm(x_orig_cont))^2;
    mse_cont_db  = gather(10 * log10(mse_cont));

    % =========================
    % Display results
    % =========================
    fprintf('✓ Recovery successful!\n');
    fprintf('MSE (discrete)      : %.2e\n', gather(mse));
    fprintf('MSE (continuous)    : %.2e\n', gather(mse_cont));
    fprintf('MSE dB (discrete)   : %.2f dB\n', mse_db);
    fprintf('MSE dB (continuous) : %.2f dB\n', mse_cont_db);
    fprintf('Computation time    : %.3f seconds\n\n', elapsed_time);
    %plot_recovery_results(x_orig_cont,x_orig, x_mod, x_rec,x_rec_cont,t_d,t_s);

end

main()
