function [x_orig_cont,t_d]=generate_original_signal(E, W,OF)
    % Parameter
    NoS = 20;        % Number of sinc components (set this as needed)
    BW = 2 * pi * W; % Bandwidth in rad/s

    % Derived parameters
    Wnyq = 2 * BW;         % Nyquist frequency
    Tnyq = 2 * pi / Wnyq;  % Nyquist sampling interval
    Ws = OF * Wnyq;        % Sampling frequency with oversampling
    Ts = 2 * pi / Ws;      % Corresponding sampling period
    Td = Ts / 1000;        % Dense time axis step for plotting

    CoS = 2*rand(NoS,1) - 1;  % Random coefficients for sinc components
    t = -1 : Td : 1;           % Time axis
    t = t(:);                  % Ensure column vector

    f = zeros(length(t), 1);   % Initialize signal

    % Construct the bandlimited signal as a sum of sinc functions
    for m = 1 : NoS
        f = f + CoS(m) * sinc(2*W * (t) - Ts*(m)*10);
    end

    % Normalize signal
    c = 1; %maximum value attained by signal
    f = c * f / max(abs(f));
    x_orig_cont=f;
    t_d=t;
end