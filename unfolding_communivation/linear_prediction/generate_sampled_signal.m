function [x_orig,t_s]=generate_sampled_signal(x_orig_cont,t_d)
    x_orig = downsample(x_orig_cont, 1000); % Sample values
    t_s = downsample(t_d, 1000); % Sampling locations
end