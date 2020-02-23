function ret = logSigDeriv(inV)
% This function evaluates the derivative of the logSigmoid function 
% at the desired value if needed 
%[rows cols] = size(inV);
%for p = 1:rows
%    ret(p, 1) = (1-logSigmoid(inV(p))) * logSigmoid(inV(p));

ret = (1-logSigmoid(inV)) .* logSigmoid(inV)
end

