function SER = symbol_error_function(a_true, bestEstimate)
% Computes:
%   error = (1/N) * sum( a_true(i) ~= bestEstimate(i) )
%
% Inputs:
%   a_true       - true transmitted symbols      (Nx1)
%   bestEstimate - estimated (or detected) symbols (Nx1)
%
% Output:
%   SER          - symbol error rate

    a_true       = a_true(:);
    bestEstimate = bestEstimate(:);

    N = length(a_true);

    SER = sum(a_true ~= bestEstimate)/N ;

    SER=SER*100;
end
