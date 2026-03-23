function N_lam = N_lambda(Delta, x_orig)

    % Indices where unfolding is required
    idx = find(abs(x_orig) > Delta/2);

    if isempty(idx)
        N_lam = 0;
        return;
    end

    Ns = length(x_orig);
    center = floor(Ns/2) + 1;

    L = idx(1);
    R = idx(end);

    % Half-support: max distance from center
    N_lam = max(abs(L - center), abs(R - center));
end
