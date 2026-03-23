function N_lam = estimate_N_lam(x_mod,h_coeffs,Delta)
    
    start=length(h_coeffs)+1;
    N_lam=start-1;
    for n=start:length(x_mod)
        x_mod_h3 = x_mod_h(x_mod,h_coeffs,n); 
        if (x_mod_h3)^2<((Delta)^2/4)
            N_lam = N_lam +1; % Increment N_lam for each valid condition
        else
            break;
        end
    end

end
