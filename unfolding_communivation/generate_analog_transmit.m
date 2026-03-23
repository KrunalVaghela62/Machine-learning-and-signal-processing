function [x_orig_cont,t_d,CoS,NoS] = generate_analog_transmit(E, W, OF, S)

    % Parameters
    NoS = 4;                % Number of sinc components
    BW = 2 * pi * W;         % Bandwidth (rad/s)
    % LCM constant (works for OF ≤ 10)
    K = fold(@lcm,2:10);
    % Fixed dense grid
    Td = 1/(2*W*K);
    % Derived parameters
    Wnyq = 2 * BW;
    Tnyq = 2 * pi / Wnyq;    
    Ws   = OF * Wnyq;
    Ts   = 2 * pi / Ws;
    %Td   = Ts / 1000;        % Dense time grid
    %Td=10e-7;
    % Coefficients for sinc amplitudes
    CoS = S(randi(numel(S), NoS, 1));
    %CoS=[1,2,1,-1];
    %CoS=[2 2 2 2 2 2 2 2 2 2]
    % Time vector (common for both plots)
    t = (-2 : Td : 2).';
    f = zeros(length(t),1);  % Sum of all sinc pulses

    % === Figure 2: individual pulses ===
    %figure; hold on;

    % Construct signal and plot individual sinc components
    for m = 1:NoS
        y_m = CoS(m) * sinc( 2*W * (t - ((2*m-(NoS))*Tnyq)));

        % accumulate for final output
        f = f + y_m;

        % plot individual sinc pulses (not added)
        %plot(t, y_m);
    end
%{
    hold off;
    xlabel('Time');
    ylabel('Individual sinc pulses');
    title('Individual CoS(m) * sinc(...) components');
    grid on;

    % === Figure 1: final summed signal ===
    figure;
    plot(t, f, 'LineWidth', 1.2);
    xlabel('Time');
    ylabel('Summed signal');
    title('Sum of all sinc components');
    grid on;
%}
    % Return outputs
    x_orig_cont = f;
    t_d = t;

end
