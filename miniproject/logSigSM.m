function [output] = logSigSM(inputV, W1, b1, W2, b2, W3, b3)
%   Function operates as a 2-layer neural network with log sigmoid transfer
%   functions for each layer. 
%   INPUTS:
%   inputV = input pattern vector
%   W1 = weight matrix for first layer
%   b1 = bias vector for first layer
%   W2 = weight matrix for second layer
%   b2 = bias vector for second layer
%   W3 = weight matrix for third layer
%   b3 = bias vector for third layer 
%   OUTPUTS:
%   output = the resulting vector outputted from the network 
output = softmax(W3 * (logSigmoid(W2 * (logSigmoid(W1 * inputV + b1)) + b2)) + b3);
end

