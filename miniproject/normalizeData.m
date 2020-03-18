function [outV] = normalizeData(inputV)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[rows, cols] = size(inputV);
outV = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        if(inputV(i, j) ~= 0)
            outV(i,j) = (inputV(i,j))/255;
        end
    end
end

end

