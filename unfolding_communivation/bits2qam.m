function sym = bits2qam(bits, M)
k = log2(M);
m = sqrt(M);
L = k/2;

bits = reshape(bits,k,[]).';

Igray = bi2de(bits(:,1:L),'left-msb');
Qgray = bi2de(bits(:,L+1:end),'left-msb');

% Gray → binary
Ibin = gray2bin(Igray);
Qbin = gray2bin(Qgray);

levels = -(m-1):2:(m-1);
Es = mean(levels.^2);
levels = levels/sqrt(2*Es);

sym = levels(Ibin+1) + 1j*levels(Qbin+1);
end

function b = gray2bin(g)
b = g;
while any(g)
    g = floor(g/2);
    b = bitxor(b,g);
end
end
