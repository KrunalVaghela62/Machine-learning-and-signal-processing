%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NO-FOLD ML DETECTION (WITH NOISE + LPF)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% ================= USER PARAMETERS =========================
OF_list = [2:8];
num_MC  = 100;

S  = [-1 1 2 -2];   % Symbol set
E  = 10;
W  = 10;

lambda = 0.6;
sigma_lpf = 0.5/1.7;

SER_no_fold = zeros(length(OF_list), num_MC);

figure; hold on; grid on;
xlabel("Oversampling Factor (OF)");
ylabel("Symbol Error Rate (SER)");

%% ================= MAIN LOOP ================================
for of_idx = 1:length(OF_list)

    OF = OF_list(of_idx);
    fprintf("\n========== OF = %d ==========\n", OF);

    %% --- Build codebook ONCE per OF ---
    [x_tmp, t_d, ~, NoS] = generate_analog_transmit(E, W, OF, S);

    idx = (t_d >= -1.8) & (t_d <= 1.8);
    t_d_crop = t_d(idx);

    [~, t_s] = generate_sampled_signal(x_tmp, t_d_crop, W, OF);

    A = generate_all_as(S, NoS);
    H = build_H_matrix(W, NoS, t_s);
    x_all = H * A;

    %% ================= MONTE CARLO ==========================
    for mc = 1:num_MC

        %% 1) Generate clean signal
        [x_cont, t_d, CoS_true, ~] = ...
            generate_analog_transmit(E, W, OF, S);

        Td = t_d(2) - t_d(1);
        Fs = 1 / Td;

        %% 2) Add noise + LPF
        sigma = sqrt((sigma_lpf^2)/(2*W*Td));
        noise = sigma * randn(length(t_d),1);

        % Low-pass filter the noise
        x_noise = lowpass(noise, W, Fs, ...
            'ImpulseResponse','iir','Steepness',0.9999);

        x_cont = x_cont + x_noise;

        %% 3) Crop & sample
        idx = (t_d >= -1.8) & (t_d <= 1.8);
        t_d = t_d(idx);
        x_cont = x_cont(idx);

        [x_samp, ~] = ...
            generate_sampled_signal(x_cont, t_d, W, OF);

        x_samp = x_samp(:);

        %% =====================================================
        % NO-FOLD ML DETECTION
        %% =====================================================
        d = sum((x_all - x_samp).^2, 1);
        [~, idx_min] = min(d);

        est = A(:, idx_min);

        SER_no_fold(of_idx, mc) = ...
            symbol_error_function(CoS_true(:), est(:));

        if mod(mc,100)==0
            fprintf("  MC %d / %d\n", mc, num_MC);
        end

    end

    %% ================= PLOT ==========================
    meanSER = mean(SER_no_fold(1:of_idx,:),2);

    plot(OF_list(1:of_idx), meanSER, '-og', ...
        'LineWidth',1.5,'MarkerFaceColor','g');

    drawnow;

end

legend("No-Fold ML (Noise + LPF)");
title("SER vs Oversampling Factor");

fprintf("\nSimulation Complete.\n");