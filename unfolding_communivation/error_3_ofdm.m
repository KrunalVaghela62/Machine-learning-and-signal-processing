%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LIVE BER vs CR (OFDM + Modulo ADC, ML detection)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% ================= PARAMETERS =================
lambda_list = 0.4:0.2:1.6;   % CR = lambda
num_MC = 500;

W   = 10;
OF  = 4;
S   = [1+1j 1-1j -1+1j -1-1j]/sqrt(2);
NoS = 5;

EbN0_dB = 20;

rng(0);

BER = zeros(size(lambda_list));

%% ================= Eb/N0 =================
Es = 1;
Eb = Es/(NoS*log2(length(S)));
EbN0 = 10^(EbN0_dB/10);
N0   = Eb/EbN0;
N0=0.01;
sigma2 = N0/2;

%% ================= LIVE FIGURE =================
figure('Color','w'); hold on; grid on;
set(gca,'YScale','log');
xlabel('Clipping Ratio (CR)');
ylabel('BER');
ylim([1e-4 1]);
xlim([lambda_list(1) lambda_list(end)]);
title(sprintf('Live BER vs CR (Eb/N0 = %d dB)',EbN0_dB));

h = plot(NaN,NaN,'-o','LineWidth',2,'MarkerSize',8);

%% ================= MONTE-CARLO =================
for idx = 1:length(lambda_list)

    lambda = lambda_list(idx);
    bit_err = 0;
    bit_tot = 0;

    for mc = 1:num_MC

        [x_cont,t_d,CoS_true] = generate_ofdm_signal(W,OF,S,NoS);
        CoS_true;
        [x_s,t_s] = generate_sampled_signal(x_cont,t_d,W,OF);
        sigma_x=sqrt(mean(abs(x_s).^2));
        x_s=x_s/sigma_x;

        Td = t_d(2)-t_d(1);
        Fs = 1/Td;
        noise = sqrt(sigma2)*(randn(size(x_s))+1j*randn(size(x_s)));
        noise = lowpass(real(noise),W,Fs) + 1j*lowpass(imag(noise),W,Fs);
        x_noisy = x_s + noise;
        
        x_mod = complex( ...
            mod(real(x_noisy)+lambda,2*lambda)-lambda, ...
            mod(imag(x_noisy)+lambda,2*lambda)-lambda );

        Pmat = build_H_matrix(W,NoS,t_s);
        Pmat = (Pmat*dftmtx(NoS)')/sqrt(NoS);
        A = generate_all_as(S,NoS);
        X_all = Pmat*A;

        dmin = inf; idx2 = 1;
        for k = 1:size(X_all,2)
            cand = X_all(:,k);
            cand=cand/sqrt(mean(abs(cand).^2));
            cand_mod = complex( ...
                mod(real(cand)+lambda,2*lambda)-lambda, ...
                mod(imag(cand)+lambda,2*lambda)-lambda );
            d = norm(x_mod-cand_mod)^2;
            if d < dmin, dmin = d; idx2 = k; end
        end

        est = A(:,idx2)';
        bits_tx = qam2bits(CoS_true,4);
        bits_rx = qam2bits(est,4);

        bit_err = bit_err + sum(bits_tx~=bits_rx);
        bit_tot = bit_tot + length(bits_tx);
    end

    BER(idx) = bit_err/bit_tot

    %% ----- LIVE UPDATE -----
    set(h,'XData',lambda_list(1:idx),'YData',BER(1:idx));
    drawnow;
end
