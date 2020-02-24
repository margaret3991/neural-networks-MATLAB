function [outputV] = validationSetTest(inputV,labels, W1, b1, W2, b2)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
[Irows, Icols] = size(inputV);
[Trows, Tcols] = size(labels);
outputV = zeros(1, Tcols);

for i = 1:Icols
    interm = zeros(Trows, 1);
    interm = logSig2Layer(inputV(:, i), W1, b1, W2, b2);
    outputV(1,i) = evaluateOutput(interm);
end

end

