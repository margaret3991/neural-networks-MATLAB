function [output] = logSig2Layer(inputV, W1, b1, W2, b2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
output = logSigmoid(W2 * (logSigmoid(W1 * inputV + b1)) + b2);
end

