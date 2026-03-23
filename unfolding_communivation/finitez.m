function next_sample()
    E  = 0.01;% Target signal energy
    lambda=0.2; %in b2r2 we use lambda as 
    Delta = 2*lambda;    % Modulo threshold
    W = 50; 
    OF=10;
    c = 1;
    BW = 2 * pi * W; % Bandwidth in rad/s
    [x_orig_cont,t_d]=generate_original_signal(E, W,OF);
    [x_orig,t_s]=generate_sampled_signal(x_orig_cont,t_d);  
    N_true=N_lambda(Delta,x_orig);
    x_mod = modulo_operation(x_orig, Delta);   
    epsilon=0.002;
    sigma=0.0125;
    noise = sigma * (2*rand(size(x_mod)) - 1); % Generate noise
    x_mod = x_mod+noise;
    % Derived parameters
    Wnyq = 2 * BW;         % Nyquist frequency
    Ws = OF * Wnyq;        % Sampling frequency with oversampling factor
    Ts = 2 * pi / Ws;
    K =compute_required_filter_length(W, E, 2*epsilon, Ts);%compute_required_filter_length(W, E, Delta, Ts);
    filter_length = 2*K;  % As per equation (9) in the paper
    disp(filter_length);
    h_coeffs = design_chebyshev_predictor(K, W, Ts);
    N_init = estimate_N_lam(x_mod,h_coeffs,Delta);
    [c_n] = possible_z_values(c,Delta);
    
    z = zeros(length(x_orig),1);
    x_rec = zeros(length(x_orig),1);
    
    
    x_rec(1:N_init)=x_mod(1:N_init);
    for n=N_init:length(x_mod)
        x_mod_h1 = x_mod_h(x_mod,h_coeffs,n);
        z(n) = z_recovery(c_n,Delta,x_mod_h1,n,z,h_coeffs);
        x_rec(n) = x_mod(n)-z(n);
    end
    fprintf('recovered first sample in presence of noise : %.2e\n',x_rec(N_init+1));
    fprintf('noise in that sample : %.2e\n',noise(N_init+1));
    fprintf('original sample : %.2e\n',x_orig(N_init+1));
    
    fprintf('Z predicted in presence of noise : %.2e\n',z(N_init+1));
%{
    z_true = zeros(length(x_orig),1);
    x_rec_true = zeros(length(x_orig),1);
    x_mod_true = modulo_operation(x_orig, Delta);
    
    x_rec_true(1:N_init)=x_mod_true(1:N_init);
    for n=N_init:length(x_mod)
        x_mod_h2 = x_mod_h(x_mod_true,h_coeffs,n);
        z_true(n) = z_recovery(c_n,Delta,x_mod_h2,n,z_true,h_coeffs);
        x_rec_true(n) = x_mod_true(n)-z_true(n);
    end
    fprintf('recovered first sample in absence of noise : %.2e\n',x_rec_true(N_init+1));
    fprintf('modulo sample : %.2e\n',x_mod_true(N_init+1));
    fprintf('Z predicted in absence of noise : %.2e\n',z_true(N_init+1));
%}
    max_diff=max(x_orig-x_rec);
    mse=(norm(x_orig - x_rec))^2/(norm(x_orig))^2;
    mse_db=10*log10(mse);
    %x_rec_cont = sinc_interpolation(x_rec, t_s, t_d, Ts,OF);
    %x_check=sinc_interpolation(x_orig, t_s, t_d, Ts,OF);
    %mse_cont = (norm(x_orig_cont - x_rec_cont))^2/(norm(x_orig_cont))^2;
    %mse_check = (norm(x_orig_cont - x_check))^2/(norm(x_orig_cont))^2;
    %mse_cont_db = 10*log10(mse_cont);
    %mse_check_db=10*log10(mse_check);
    fprintf('max differnce between samples : %.2e\n',max_diff);
    fprintf('MSE_newapproch: %.2e\n', mse);
    fprintf('MSE_newapproch in dbs: %.2e\n', mse_db);
    fprintf('N_lamda value: %.d\n', N_init);
    fprintf('N_lamda TRUE value: %.d\n', N_true);
    %fprintf('MSE_cont_new_approach: %.2e\n', mse_cont);
    %fprintf('MSE_db_cont_new_approach: %.2e\n', mse_cont_db);
    %fprintf('MSE_db_check: %.2e\n', mse_check);
    %fprintf('MSE_db_cont_check: %.2e\n', mse_check_db);
    
  %{
    fig4 = figure('Name', 'original and recovered Signal', ...
                  'Units', 'normalized', ...
                  'OuterPosition', [0 0 1 1]);
    hold on;
    %plot(t_s, x_rec, 'r', 'LineWidth', 3);
    stem(t_s, x_orig, 'r', 'LineWidth', 3);
    stem(t_s, x_rec, 'b', 'LineWidth', 3);
    xlabel('$t$','FontSize',30,'FontName','Times','Interpreter','Latex')
    
    set(gca,'FontSize',30,'FontName','Times')
    % Optionally add waitfor(fig2) if you want to block until closed
    %% Plot 3: modulo Signal
  %}
    %{
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
    
    %}
end

next_sample()