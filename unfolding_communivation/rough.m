%% --- Ensure t_s is exactly the downsample of t_d used earlier ---
% (use the same downsample rule as your generate_sampled_signal)
ds_factor = 1000;
t_s = t_d(1:ds_factor:end);    % explicit, guaranteed match to generate_sampled_signal
[x_orig_check, ~] = generate_sampled_signal(x_filt, t_d);  % just to confirm lengths
CoS
% Sanity check: lengths must match
if length(t_s) ~= length(x_orig_check)
    error('t_s length mismatch: build t_s consistently with generate_sampled_signal.');
end

%% --- Build H using the same t_s ---
H = build_H_matrix(W, NoS, t_s);   % Ns x NoS

%% --- Generate all symbol sequences A (NoS x NumComb) ---
A = generate_all_as(S, NoS);       % columns are sequences
% (If generate_all_as returns NoS x NumComb, this is correct for H*A)

%% --- Build all candidate sampled signals: x_orig_all (Ns x NumComb) ---
x_orig_all = H * A;   % Ns x NumComb

%% --- Apply modulo to all columns (vectorized) ---
% Make sure modulo_operation is vectorized (it already is if written as x - Delta*round(x/Delta))
x_mod_all = modulo_operation(x_orig_all, Delta);   % Ns x NumComb

%% --- Observed folded signal x_mod (Ns x 1) must match x_mod_all rows ---
% Recompute x_mod from sampled x_orig to be safe:
x_orig = generate_sampled_signal(x_filt, t_d);  % returns vector
x_mod = modulo_operation(x_orig, Delta);

% Confirm sizes
[Ns, NumComb] = size(x_mod_all);
if length(x_mod) ~= Ns
    error('Length mismatch even after construction: length(x_mod)=%d, rows(x_mod_all)=%d', length(x_mod), Ns);
end

%% --- Compute modulo-aware wrapped error and find best match (vectorized) ---
% Compute elementwise wrapped difference into [-Delta/2, Delta/2)
E = mod( x_mod(:,ones(1,NumComb)) - x_mod_all + Delta/2, Delta ) - Delta/2;  % Ns x NumComb
dists = sqrt( sum( E.^2, 1 ) );    % 1 x NumComb
[best_dist, best_idx] = min(dists);

% Result
best_symbols = A(:, best_idx).';
fprintf('Best index = %d, best dist = %g\n', best_idx, best_dist);
disp('Recovered symbol vector (best_symbols):');
disp(best_symbols);
