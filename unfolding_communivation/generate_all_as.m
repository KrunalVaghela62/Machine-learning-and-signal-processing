function A = generate_all_as(S, N)
% Generates all vectors of length N with values taken from S
% Output A has size (N x |S|^N), each column is one vector.

    % Create 1×N cell array where each entry is the symbol set S
    vecs = repmat({S}, 1, N);

    % Generate all combinations
    A = combvec(vecs{:});
end
