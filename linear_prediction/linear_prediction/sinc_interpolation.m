function x_cont = sinc_interpolation(x_sampled, t_s, t_d, Ts, OF)
% SINC_INTERPOLATION  Memory-safe Shannon interpolation
%
%   x_cont = sinc_interpolation(x_sampled, t_s, t_d, Ts, OF)
%
% Inputs:
%   x_sampled - [Ns x 1] sampled signal values
%   t_s       - [Ns x 1] time instances of samples (n*Ts)
%   t_d       - [1 x Nd] desired time points for reconstruction
%   Ts        - sampling period
%   OF        - oversampling factor
%
% Output:
%   x_cont    - [Nd x 1] reconstructed continuous-time signal

    % Ensure column vectors
    x_sampled = x_sampled(:);
    t_s       = t_s(:);
    Nd        = numel(t_d);

    % Preallocate
    x_cont = zeros(Nd,1);

    % Choose a block size (tune this depending on RAM, e.g., 1e4 points at a time)
    block_size = 1e4;

    % Process in chunks of t_d
    for k = 1:block_size:Nd
        idx = k:min(k+block_size-1, Nd);   % block indices
        td_block = t_d(idx);              % current subset of desired times

        % Compute sinc kernel only for this block
        arg = (td_block(:).' - t_s) ./ (Ts);   % Ns x block_size
        S   = sinc(arg);

        % Reconstruct for this block
        x_cont(idx) = (x_sampled.' * S).';
    end
end
