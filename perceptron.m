function [a] = perceptron(W, p, b)
%	Function models a single neuron perceptron
%   with the hardlim transfer function

%   W = weight matrix 
%   p = input/pattern vector
%   b = bias vector 

a = hardlim(W * p + b);
end

