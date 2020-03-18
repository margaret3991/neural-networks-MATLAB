function [output] = ReLU2Layer(inputV, W1, b1, W2, b2)
%   Function operates as a 2-layer neural network with log sigmoid transfer
%   functions for each layer. 
%   INPUTS:
%   inputV = input pattern vector
%   W1 = weight matrix for first layer
%   b1 = bias vector for first layer
%   W2 = weight matrix for second layer
%   b2 = bias vector for second layer
%   OUTPUTS:
%   output = the resulting vector outputted from the network 
output = ReLU(W2 * (ReLU(W1 * inputV + b1)) + b2);
end

