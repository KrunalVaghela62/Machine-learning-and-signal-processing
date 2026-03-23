function H = build_H_matrix(W, NoS, t_s)

    % Nyquist spacing (since sinc uses 2W)
    Tnyq = 1/(2*W);

    Ns = length(t_s);
    H  = zeros(Ns, NoS);

    for m = 1:NoS
        shift = (2*m - NoS) * Tnyq;
        H(:, m) = sinc( 2*W * (t_s - shift) );
    end
end
