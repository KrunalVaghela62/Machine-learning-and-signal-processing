clc; clear; close all;

%% ================= PARAMETERS =================
W   = 10;
S   = [1 2 -1 -2];
NoS = 5;
OF  = 4;
num_MC = 200;

lambda = 0.2;                     % FIXED modulo threshold
sigma_list = lambda./(10:-1:1);  % lambda/10 ... lambda

SER_ML = zeros(length(sigma_list),1);
rng(0);
SER_frame = zeros(num_MC,1);

%% ================= LIVE PLOT =================
figure('Color','w'); grid on; hold on;
xlabel('\sigma');
ylabel('Symbol Error Rate (SER)');
title('SER vs Noise Level (Fixed \lambda = 0.2, OF = 4)');
h = plot(NaN,NaN,'-ob','LineWidth',1.8,'MarkerFaceColor','b');

%% ================= MONTE CARLO =================
for i = 1:length(sigma_list)

    sigma = sigma_list(i);
    fprintf('sigma = %.4f (%d/%d)\n', sigma, i, length(sigma_list));

    for mc = 1:num_MC

        %% ----- Generate OFDM -----
        [x_cont, t_d, CoS_true] = generate_ofdm_signal(W,OF,S,NoS);

        %% ----- Add noise -----
        Td = t_d(2)-t_d(1);
        Fs = 1/Td;

        noise = sigma*(randn(size(x_cont))+1j*randn(size(x_cont)));
        noise = lowpass(real(noise),W,Fs) + 1j*lowpass(imag(noise),W,Fs);
        x_noisy = x_cont + noise;

        %% ----- Sample -----
        [x_s, t_s] = generate_sampled_signal(x_noisy,t_d,W,OF);
        x_true = x_s(:);

        %% ----- Modulo ADC (fixed lambda) -----
        x_mod = complex( ...
            mod(real(x_true)+lambda,2*lambda)-lambda, ...
            mod(imag(x_true)+lambda,2*lambda)-lambda );

        %% ----- Dictionary -----
        Pmat = build_H_matrix(W,NoS,t_s);
        Pmat = Pmat/sqrt(NoS);

        A = generate_all_as(S,NoS);
        X_all = Pmat*(dftmtx(NoS)')*A;

        %% ----- Exhaustive ML -----
        dmin = inf; idx = 1;
        for k = 1:size(X_all,2)
            cand = X_all(:,k);
            cand_mod = complex( ...
                mod(real(cand)+lambda,2*lambda)-lambda, ...
                mod(imag(cand)+lambda,2*lambda)-lambda );
            d = norm(x_mod - cand_mod)^2;
            if d < dmin
                dmin = d; idx = k;
            end
        end

        estML = A(:,idx);

        %% ----- SER -----
        SER_frame(mc) = symbol_error_function(CoS_true, estML);

    end

    SER_ML(i) = mean(SER_frame);

    %% ===== LIVE UPDATE =====
    set(h,'XData',sigma_list(1:i),'YData',SER_ML(1:i));
    drawnow;
end
