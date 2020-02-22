function output = layer(p, W, b)
% Multi-input neuron layer function 
%   a = f(wp + b)
%   W is matrix, p and b are vectors 
[rows, cols] = size(W)
% rows = number of neurons needed 
output = hardlim(W * p' + b)
end
