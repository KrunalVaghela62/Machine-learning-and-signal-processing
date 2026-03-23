function K = compute_required_filter_length(W, E, Delta, Ts)
    numerator = log(sqrt(32*W*E) / Delta);
    denominator = log(2 / (1 - cos(2*pi*W*Ts)));
    K = ceil(numerator / denominator);
    K = max(K, 1);
end