function montecarlo_mse_db_vs_of_gpu()
    % Parameters
    E = 50;
    lambda = 0.1;
    Delta = 2 * lambda;
    W = 50;
    OFs = [10 15 20 25 30];
    num_trials = 5;
    snrs = [30, 40, 50];
    
    % Preallocate results
    mean_mse_db = zeros(length(snrs), length(OFs));
    std_mse_db = zeros(length(snrs), length(OFs));
    
    for idx_snr = 1:length(snrs)
        snr = snrs(idx_snr);
        fprintf('\n===== Monte Carlo simulation for SNR = %d dB =====\n', snr);
        
        mse_db_trials = zeros(num_trials, length(OFs), 'gpuArray');
        
        for idx_of = 1:length(OFs)
            OF = OFs(idx_of);
            fprintf('  --> Running for Oversampling Factor = %d (%d/%d)\n', OF, idx_of, length(OFs));
            
            for trial = 1:num_trials
                % -- Signal generation --
                [x_orig_cont, t_d] = generate_original_signal(E, W, OF);
                [x_orig_cpu, t_s] = generate_sampled_signal(x_orig_cont, t_d);
                x_orig = gpuArray(x_orig_cpu);
                t_s = gpuArray(t_s);
                t_d = gpuArray(t_d);
                
                % -- Modulo and noise addition --
                x_mod = modulo_operation(x_orig, Delta);
                noise_bound = lambda * 10^(-snr/20);
                noise = (2 * rand(size(x_mod), 'gpuArray') - 1) * noise_bound;
                x_mod = x_mod + noise;
                
                % -- Recovery --
                Ts = t_s(2) - t_s(1);
                [x_rec, success] = modulo_recovery_corrected(x_mod, Ts, W, Delta, E, x_orig);
                x_rec_cont = sinc_interpolation(x_rec, t_s, t_d, Ts, OF);
                
                % -- Error metric (continuous) --
                mse_cont = (norm(x_orig_cont - x_rec_cont))^2 / (norm(x_orig_cont))^2;
                mse_db_trials(trial, idx_of) = 10 * log10(mse_cont);
            end
        end
        
        % Gather to CPU, compute stats
        mse_db_trials_cpu = gather(mse_db_trials);
        mean_mse_db(idx_snr, :) = mean(mse_db_trials_cpu, 1);
        std_mse_db(idx_snr, :) = std(mse_db_trials_cpu, 0, 1);

        % Plot -- big window for each SNR
        figure('Name', sprintf('MSE (dB) vs OF for SNR %d dB', snr), 'Position', [100 100 1200 600]);
        errorbar(OFs, mean_mse_db(idx_snr, :), std_mse_db(idx_snr, :), 'o-', ...
                 'LineWidth', 2, 'MarkerSize', 8);
        xlabel('Oversampling Factor (OF)');
        ylabel('Mean Continuous MSE (dB)');
        title(sprintf('Corrected Recovery: MSE (dB) vs OF, SNR = %d dB', snr));
        grid on;
        set(gca, 'FontSize', 16);
    end
end

montecarlo_mse_db_vs_of_gpu();
