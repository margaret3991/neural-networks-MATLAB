function [targetOut] = evaluateOutput(inputV)
[rows cols] = size(inputV);
largestIndex = 1;
largestVal = inputV(1,1);
for r = 1:rows
    if(inputV(r, 1) > largestVal)
        largestIndex = r;
        largestVal = inputV(r,1);
    end
end
targetOut = largestIndex - 1;
    
end

