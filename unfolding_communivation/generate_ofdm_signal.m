function [x_ofdm_cont, t_d, CoS, NoS] = generate_ofdm_signal(W, OF, S, M)
% ---------------------------------------------------------
% Time-domain OFDM generation (RECT windowed)
% Subcarrier indexing: (2k - (N+1)) * (W/N)
% Produces sinc-shaped spectrum (verified via FFT)
% ---------------------------------------------------------

    %% ================= SYMBOL PARAMETERS =================
    NoS = M;
    %Delta_f = W / NoS;
    CoS = S(randi(numel(S), NoS, 1));
    %CoS=[-2     2    -2     1     1];
    %% ================= TIME GRID =================
    BW   = 2 * pi * W;
    Wnyq = 2 * BW;
    Ws   = OF * Wnyq;

    Ts = 2*pi / Ws;
    %Td = Ts / 1000;


    Td=1e-6;
    % Observation window (FFT support)
    Tobs = 2;
    t_d = (-Tobs/2 : Td : Tobs/2).';
    b=(dftmtx(NoS))'*CoS';
    b=b/sqrt(NoS);   
    f = zeros(length(t_d),1);  % Sum of all sinc pulses

    Tnyq = 2 * pi / Wnyq; 
    for m = 1:NoS
        y_m = b(m) * sinc( 2*W * (t_d - ((2*m-(NoS))*Tnyq)));

        % accumulate for final output
        f = f + y_m;

        % plot individual sinc pulses (not added)
        %plot(t, y_m);
    end
    x_ofdm_cont=f;

    %{
    %% ================= RECTANGULAR WINDOW =================
    rect_t = abs(t_d) <= T/2;

    %% ================= TIME-DOMAIN OFDM =================
    x_ofdm_cont = zeros(size(t_d));

    % EXACT notebook indexing
    % k = 1,...,N
    % fk = (W/N) * (2k - (N+1))
    for k = 1:NoS
        mk = 2*k - (NoS + 1);           % centered index
        fk = Delta_f * mk;              % subcarrier frequency
        
        x_ofdm_cont = x_ofdm_cont + ...
                    (2*W/NoS)*CoS(k) .* exp(1j*2*pi*fk*t_d);
       %{
        x_ofdm_cont = x_ofdm_cont + ...
                    (2*W/NoS)*CoS(k) .* exp(1j*2*pi*fk*t_d);
       %}
    end

    % Apply time window (creates sinc spectrum)
    %x_ofdm_cont = x_ofdm_cont .* rect_t;
    hehe=2*W/NoS;
    x_ofdm_cont = x_ofdm_cont .* (sinc(hehe*t_d));
    %% ================= PLOTS =================
    %}
    
%{    
     figure;

    % ---- Time-domain ----
    plot(t_d, real(x_ofdm_cont), 'b', ...
         t_d, imag(x_ofdm_cont), 'r');
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('OFDM Signal (Time Domain)');
    legend('Real','Imag');
    grid on;

    % ---- Frequency-domain ----
    subplot(2,1,2);
    Nfft = length(t_d);
    X_fft = fftshift(fft(x_ofdm_cont))*Td;
    dt = t_d(2) - t_d(1);
    f = (-Nfft/2:Nfft/2-1)/(Nfft*dt);

    plot(f, abs(X_fft), 'b', 'LineWidth', 1.5);
    xlabel('Frequency (Hz)');
    ylabel('|X(f)|');
    title('OFDM Spectrum (Sinc-shaped)');
    grid on;
    xlim([-W W]);
%}
    % Mark subcarrier centers
    
    
    
end
