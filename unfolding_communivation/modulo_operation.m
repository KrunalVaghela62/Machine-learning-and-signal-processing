function result = modulo_operation(x, Delta)
% MODULO_OPERATION Implements modulo operation as defined in the paper
% x* = [x] mod Delta as the unique number in [-Delta/2, Delta/2)
    lambda=Delta/2;
    result = mod(x + lambda, 2*lambda) - lambda;    
    % Ensure result is in [-Delta/2, Delta/2)
    if result >= Delta/2
        result = result - Delta;
    elseif result < -Delta/2
        result = result + Delta;
    end
end