function output = layerf(p, W, b)
% Multi-input neuron layer function 
%   a = f(wp + b)
%   W is matrix, p and b are vectors 
[rows, cols] = size(W)
% rows = number of neurons needed 
for i = 1:rows
    output(i) = hardlim(W * p' + b(i))
end
