function [outputV] = validTest2LaySM(inputV,labels, W1, b1, W2, b2)
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
%   W3 = weight matrix for third layer
%   b3 = bias vector for third layer 
%   OUTPUTS:
%   outputV = vector of output labels created by the network 
[Irows, Icols] = size(inputV);
[Trows, Tcols] = size(labels);
outputV = zeros(1, Tcols);
interm = zeros(Trows, 1);

for i = 1:Icols
    interm = softmax(W2 * (logSigmoid(W1 * inputV(:, i) + b1)) + b2);
    %interm = logSigSM(inputV(:, i), W1, b1, W2, b2);
    outputV(1,i) = evaluateOutput(interm);
end
end
