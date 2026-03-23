function plot_spectrum(x, t)
% plot_spectrum(x, t)
% ---------------------------------------------
% Plots the magnitude spectrum |X(f)| of a signal x(t)
% using sqrt(N) normalization.
%
% INPUTS:
%   x : signal values (vector)
%   t : corresponding time samples (vector)
%
% This function computes:
%   - sampling frequency
%   - FFT with root-N normalization
%   - correct frequency axis in Hz
%   - magnitude spectrum plot
% ---------------------------------------------

    % Ensure column vectors
    x = x(:);
    t = t(:);

    % Sampling information
    Td = t(2) - t(1);     % time step
    Fs = 1 / Td;          % sampling frequency (Hz)
    N  = length(x);       % number of samples

    % FFT with sqrt(N) normalization
    X = fftshift( fft(x) / N) ;

    % Frequency axis
    f_axis = linspace(-Fs/2, Fs/2, N);

    % Plot spectrum
    figure;
    plot(f_axis, abs(X), 'LineWidth', 1.2);
    xlabel('Frequency (Hz)');
    ylabel('|X(f)|');
    title('Magnitude Spectrum');
    grid on;

end
