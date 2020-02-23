function ret = logSigDeriv(num)
% This function evaluates the derivative of the logSigmoid function 
% at the desired value if needed 

ret = (1-logSigmoid(num)) * logSigmoid(num);
end

