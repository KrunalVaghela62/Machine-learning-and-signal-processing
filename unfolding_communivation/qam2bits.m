function bits = qam2bits(sym, M)
% Works for M = 4,16,64,256,...
% Quadrant-consistent Gray mapping

k = log2(M);
m = sqrt(M);          % points per axis
L = k/2;              % bits per axis

% PAM levels
levels = -(m-1):2:(m-1);

% normalize to unit average power
Es = mean(levels.^2);
levels = levels/sqrt(2*Es);

% Slice
[~,Iind] = min(abs(real(sym) - levels.'),[],1);
[~,Qind] = min(abs(imag(sym) - levels.'),[],1);

% Binary indices (0...m-1)
Ibin = Iind-1;
Qbin = Qind-1;

% Binary -> Gray (axis-wise)
Igray = bitxor(Ibin, floor(Ibin/2));
Qgray = bitxor(Qbin, floor(Qbin/2));

% Convert to bits
Ibits = de2bi(Igray, L, 'left-msb');
Qbits = de2bi(Qgray, L, 'left-msb');

bits = reshape([Ibits Qbits].',1,[]);
end
