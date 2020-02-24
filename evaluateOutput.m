function [targetOut] = evaluateOutput(inputV)
%   This function takes in the vector output of the logsigmoid 2-layer 
%   network and converts to integer classification based on the values 
%   INPUT:
%   inputV = vector of network output (from validation set)
%   OUTPUT:
%   targetOut = integer classification from 0-9

[rows cols] = size(inputV); % get size of input vector 
largestIndex = 1;
largestVal = inputV(1,1); % set highest value initially to first value
for r = 1:rows
    if(inputV(r, 1) > largestVal)   % if bigger,
        largestIndex = r;           % update the largest row/index 
        largestVal = inputV(r,1);   % update the corresponding value 
    end
end
targetOut = largestIndex - 1; % adjust for 0-9 digit pattern 
end

