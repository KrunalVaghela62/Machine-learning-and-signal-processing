function result = modulo_operation(x, Delta)
% MODULO_OPERATION Implements modulo operation as defined in the paper
% x* = [x] mod Delta as the unique number in [-Delta/2, Delta/2)
    
    result = x - Delta * round(x / Delta);
    
    % Ensure result is in [-Delta/2, Delta/2)
    if result >= Delta/2
        result = result - Delta;
    elseif result < -Delta/2
        result = result + Delta;
    end
end