%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Monte-Carlo SER vs OF
%   Exhaustive ML in the modulo domain ONLY (vectorized modulo op)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% ================= USER PARAMETERS ======================================
OF_list = 2:8;
numMC   = 500;

S       = [-1 1 2 -2];     % symbol alphabet
NoS=5;            % number of coefficients
W       = 10;
E       = 10;
SNR_dB  = 20;

Delta   = 0.4;             % full-width
Lambda  = Delta/2;

fprintf("\nRunning modulo-only ML SER simulation (vectorized modulo)...\n");

SER_vals = zeros(length(OF_list), numMC);

%% ----------- Create LIVE PLOT window -----------------
figure('Color','w','Position',[300 200 800 400]);
hold on; grid on;
xlabel("Oversampling Factor (OF)");
ylabel("SER");
title("SER vs OF (Modulo-Domain ML Only)");
set(gca,'FontSize',12);
ylim([0 1]);

h_plot = [];   % handle for updating plot

%% ================= Monte-Carlo Loop ====================================
for oi = 1:length(OF_list)

    OF = OF_list(oi);
    fprintf("\n===========================\n");
    fprintf("   OF = %d\n", OF);
    fprintf("===========================\n");

    %% Precompute codebook A (all symbol combinations)
    A = generate_all_as(S, NoS);

    for mc = 1:numMC

        %% ---- 1) Generate random transmitted symbols ----

        %% ---- 2) Generate continuous bandlimited signal ----
        [x_cont, t_d, CoS_true, NoS] = generate_analog_transmit(E, W, OF, S);

        %% ---- 3) Add noise ----
        noisy_x = awgn(x_cont, SNR_dB, 'measured');

        %% ---- 4) Ideal LPF ----
        x_filt = ideal_lowpass(noisy_x, t_d, W);

        %% ---- 5) Sample ----
        [x_samp, t_s] = generate_sampled_signal(x_filt, t_d);
        x_samp = x_samp(:);

        %% ---- 6) Apply modulo (vectorized) ----
        x_mod = modulo_operation(x_samp, Delta);

        %% ---- 7) Build H matrix ----
        H = build_H_matrix(W, NoS, t_s);

        %% ---- 8) Predicted signals for each candidate (vectorized) ----
        Xpred = H * A;

        %% ---- 9) Apply modulo to each column ----
        Xpred_mod = modulo_operation(Xpred, Delta);

        %% ---- 10) ML decision ----
        diffs = Xpred_mod - x_mod;
        d2 = sum(diffs.^2, 1);

        [~, idx_est] = min(d2);
        est_CoS = A(:, idx_est);

        %% ---- 11) SER ----
        SER_vals(oi, mc) = symbol_error_function(CoS_true, est_CoS);

        if mod(mc, 50)==0
            fprintf("   MC %d/%d\n", mc, numMC);
        end
    end

    %% ============================================================
    %      UPDATE LIVE PLOT AFTER EACH OF IS COMPLETED
    %% ============================================================
    SER_mean_partial = mean(SER_vals(1:oi, :), 2);
    SER_std_partial  = std(SER_vals(1:oi, :), 0, 2);

    % Remove old plot handle
    if ~isempty(h_plot)
        delete(h_plot);
    end

    % Draw updated curve
    h_plot = errorbar(OF_list(1:oi), SER_mean_partial, SER_std_partial, ...
                      '-ob','LineWidth',1.5,'MarkerSize',7,'MarkerFaceColor','b');

    drawnow;   % <-- forces the plot to refresh immediately

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            SUPPORT FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function x_filt = ideal_lowpass(x, t, cutoff_hz)
    Td = t(2) - t(1);
    Fs = 1 / Td;
    N  = length(x);
    X  = fftshift(fft(x));
    f_axis = linspace(-Fs/2, Fs/2, N).';
    H = abs(f_axis) <= cutoff_hz;
    Xf = X .* H;
    x_filt = real(ifft(ifftshift(Xf)));
end



