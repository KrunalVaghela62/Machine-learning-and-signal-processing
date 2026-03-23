function P = build_P_ofdm_matrix(t_s, W, NoS)
% ---------------------------------------------------------
% P-matrix EXACTLY matching generate_ofdm_signal
%
% x_s = P * a
%
% Includes:
% - (2k-(N+1)) indexing
% - (W/N) amplitude scaling
% - rectangular window in time
% ---------------------------------------------------------

    Delta_f = W / NoS;
    T       = NoS / W;

    Ns = length(t_s);
    P  = zeros(Ns, NoS);

    % Rectangular window (same as generator)
    rect_t = abs(t_s) <= T/2;
    hehe=2*W/NoS;
    for k = 1:NoS
        mk = 2*k - (NoS + 1);
        fk = Delta_f * mk;
        %{
        P(:,k) = (W/NoS) ...
               .* exp(1j*2*pi*fk*t_s) ...
               .* rect_t;
         %}
        P(:,k) = (2*W/NoS) ...
               .* exp(1j*2*pi*fk*t_s) ...
               .* (sinc(hehe*t_s));
       
    end
end
