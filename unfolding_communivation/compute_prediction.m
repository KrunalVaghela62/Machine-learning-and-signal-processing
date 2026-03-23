function x_pred = compute_prediction(x_recovered, n, h_coeffs, filter_length)
    x_pred = 0;
    l=length(h_coeffs);
    for i = 1:min(l, n - 1)
        x_pred = x_pred + h_coeffs(l-i+1) * (x_recovered(n - i));
    end
end