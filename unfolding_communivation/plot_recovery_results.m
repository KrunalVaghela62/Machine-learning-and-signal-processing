function plot_recovery_results(x_orig_cont,x_orig, x_mod, x_rec,x_rec_cont,t_d,t_s)
% PLOT_RECOVERY_RESULTS
% First shows Original + Modulo signals together,
    % then shows Recovered signal separately.
    
    
    %% Plot 1: Magnitude Spectrum of Prediction Filter h(n)
    fig6 = figure('Name','Spectrum of h(n)', ...
           'Units','normalized','OuterPosition',[0 0.5 1 0.5]);
    
    N_fft = 2048;  % Higher FFT points for better resolution
    E     = 50;    % Target signal energy
    lambda= 0.125;
    Delta = 2 * lambda;    % Modulo threshold
    W     = 50; 
    OF    = 2;
    BW    = 2 * pi * W;    % Bandwidth in rad/s
    
    % Derived parameters
    Wnyq = 2 * BW;         % Nyquist frequency
    Ws = OF * Wnyq;        % Sampling frequency with oversampling factor
    Ts = 2 * pi / Ws;
    
    % Frequency axis
    f = linspace(-1/(2*Ts), 1/(2*Ts), N_fft);
    K = compute_required_filter_length(W, E, Delta, Ts);
    h_coeffs = design_chebyshev_predictor(K, W, Ts);
    hold on;
    % FFT of h(n) — zero-padded
    h_padded = zeros(1, N_fft);
    h_padded(1:length(h_coeffs)) = h_coeffs;
    H = fftshift(fft(h_padded));
    x_padded = zeros(1, N_fft);
    x_padded(1:length(x_orig)) = x_orig;
    X = fftshift(fft(x_padded));
    
    X = X(:).';
    f = f(:).';
    H = H(:).';
    
    % Plot h(n) spectrum
    plot(f, abs(H)/abs(max(H)), 'b', 'LineWidth', 4);
    plot(f, abs(X)/abs(max(X)), 'r--', 'LineWidth', 2);
    xlabel('Frequency (Hz)', 'FontSize', 24, 'Interpreter', 'latex');
    ylabel('Magnitude', 'FontSize', 24, 'Interpreter', 'latex');
    title('Magnitude Spectrum of Prediction Filter h(n)', ...
          'FontSize', 28, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'FontName', 'Times');
    
    



    fig5=figure('Name', 'Original vs Reconstructed Continuous Signal', ...
       'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    hold on;
    plot(t_d, x_orig_cont, 'b', 'LineWidth', 4);
    plot(t_d, x_rec_cont, 'r--', 'LineWidth', 2);
    legend('Original', 'Reconstructed');
    xlabel('$t$', 'FontSize', 30,'FontName','Times', 'Interpreter', 'latex');
    title('Whittaker-Shannon Interpolation of Recovered Signal');
    set(gca,'FontSize',30,'FontName','Times')
    
    %% Plot 4: recovered Signal
    fig4 = figure('Name', 'original and recovered Signal', ...
                  'Units', 'normalized', ...
                  'OuterPosition', [0 0 1 1]);
    hold on;
    plot(t_d, x_orig_cont, 'b','LineWidth',4);
    %plot(t_s, x_rec, 'r', 'LineWidth', 3);
    stem(t_s, x_rec, 'r', 'LineWidth', 3);

    xlabel('$t$','FontSize',30,'FontName','Times','Interpreter','Latex')
    
    set(gca,'FontSize',30,'FontName','Times')
    % Optionally add waitfor(fig2) if you want to block until closed
    %% Plot 3: modulo Signal
    fig3 = figure('Name', 'original and modulo Signal', ...
                  'Units', 'normalized', ...
                  'OuterPosition', [0 0 1 1]);
    hold on;
    plot(t_d, x_orig_cont, 'b','LineWidth',4);
    stem(t_s, x_mod, 'r', 'LineWidth', 3);

    xlabel('$t$','FontSize',30,'FontName','Times','Interpreter','Latex')
    
    set(gca,'FontSize',30,'FontName','Times')
    % Optionally add waitfor(fig2) if you want to block until closed
    %% Plot 2: sampled Signal
    fig2 = figure('Name', 'sampled Signal', ...
                  'Units', 'normalized', ...
                  'OuterPosition', [0 0 1 1]);
    hold on;
    plot(t_d, x_orig_cont, 'b','LineWidth',4);
    stem(t_s, x_orig, 'r', 'LineWidth', 3);

    xlabel('$t$','FontSize',30,'FontName','Times','Interpreter','Latex')
    
    set(gca,'FontSize',30,'FontName','Times')
    % Optionally add waitfor(fig2) if you want to block until closed
    %% Plot 1: Original Signal
    fig1 = figure('Name', 'Original signal', ...
                  'Units', 'normalized', ...
                  'OuterPosition', [0 0 1 1]);  % Fullscreen
    
    hold on;
    plot(t_d, x_orig_cont, 'b', 'LineWidth', 4);
    xlabel('$t$', 'FontSize', 30, 'FontName', 'Times', 'Interpreter', 'latex')
    set(gca, 'FontSize', 30, 'FontName', 'Times')  % check
    %waitfor(fig1);  % Wait for user to close before proceeding
end
