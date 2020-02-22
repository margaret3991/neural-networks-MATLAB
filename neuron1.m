function output = neuron1(input, weight, bias)
% Single neuron function 
%   a = f(wp + b)
output = hardlim(input * weight + bias)
end

