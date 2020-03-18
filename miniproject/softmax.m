function ret = softmax(inV)
%	Log Sigmoid function for NN Transfer Functions 
%   Accepts a column vector of inputs and calculates output
[rows cols] = size(inV);
total = 0;
for p = 1:rows
    ret(p, cols) = exp(inV(p,cols));
    total = total + exp(inV(p, cols));
end
for p = 1:rows
    ret(p, cols) = ret(p,cols)/total;
end
% ret = 1./(1 + exp(-inV));
    
%ret = ret';    
end
%disp(ret)