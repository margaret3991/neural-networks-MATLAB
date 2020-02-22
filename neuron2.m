function output = neuron2(p, w, bias)
% Multi-input single neuron function 
%   a = f(wp + b)
%   W, p are vectors, bias is scalar
output = hardlim(w * p' + bias)
end
