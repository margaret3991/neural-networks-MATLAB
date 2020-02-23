function ret = logSigmoid(inV)
%	Log Sigmoid function for NN Transfer Functions 
%   Accepts a column vector of inputs and calculates output
[rows cols] = size(inV);
for p = 1:rows
    ret(p) = 1/(1 + exp(-1 * inV(p)));
ret = ret'    
end
