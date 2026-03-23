function [x_orig,t_s] = generate_sampled_signal(x_orig_cont,t_d,W,OF)
%{
    x_orig = downsample(x_orig_cont, 1000); % Sample values
    t_s = downsample(t_d, 1000); % Sampling locations
    
    BW   = 2 * pi * W;
    Wnyq = 2 * BW;
    Ws   = OF * Wnyq;

    Ts = 2*pi / Ws;
        % ================= Time resolution =====================
    dt = t_d(2) - t_d(1);          % ~ 1e-6
    step = round(Ts / dt);         % samples per Ts

    % ================= Sampling ============================
    idx = 1 : step : length(t_d);

    x_orig = x_orig_cont(idx);
    t_s    = t_d(idx);
%}
% LCM constant
    K = fold(@lcm,2:10);

    % Downsampling factor
    alpha = K/OF;

    % Downsample
    x_orig = downsample(x_orig_cont, alpha);
    t_s    = downsample(t_d, alpha);
end
