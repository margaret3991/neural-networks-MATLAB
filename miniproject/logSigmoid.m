function ret = logSigmoid(inV)
%	Log Sigmoid function for NN Transfer Functions 
%   Accepts a column vector of inputs and calculates output
%[rows cols] = size(inV);
%for p = 1:rows

ret = 1./(1 + exp(-inV));
    
%ret = ret';    
end
%disp(ret)
