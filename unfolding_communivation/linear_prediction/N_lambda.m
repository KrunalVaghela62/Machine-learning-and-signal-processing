function N_lam=N_lambda(Delta,x_orig)
    N_lam=0;
    for i = 1:length(x_orig)
        if x_orig(i)>Delta/2 || x_orig(i)<-Delta/2
            N_lam=i;
            break;
        end
    end
end