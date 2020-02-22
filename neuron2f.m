function output = neuron2f(p, w, bias)
% Multi-input single neuron function 
%   a = f(wp + b)
%   W, p are vectors, bias is scalar
for i = 1:length(p)
    dotprod += p(i) * w(i)
output = hardlim(dotprod + bias)
end


