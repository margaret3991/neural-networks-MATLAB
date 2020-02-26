function [outputV] = validationSetTest3Lay(inputV,labels, W1, b1, W2, b2, W3, b3)
%	This function receives the test inputs and targets to be run through 
%   the neural network. Calculates label for each column of the input and 
%   target matrices and outputs them as a vector 

%   INPUTS:
%   inputV = matrix of test inputs
%   labels = matrix of test targets/labels 
%   W1 = weight matrix for first layer of the network
%   b1 = bias vector for the first layer 
%   W2 = weight matrix for second layer of the network
%   b2 = bias vector for the second layer 
%   OUTPUTS:
%   outputV = vector of output labels created by the network 
[Irows, Icols] = size(inputV);
[Trows, Tcols] = size(labels);
outputV = zeros(1, Tcols);

for i = 1:Icols
    interm = zeros(Trows, 1);
    interm = logSig3Layer(inputV(:, i), W1, b1, W2, b2, W3, b3);
    outputV(1,i) = evaluateOutput(interm);
end
end

