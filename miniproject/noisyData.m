function [outV] = noisyData(inputV)

[rows, cols] = size(inputV);

for i = 1:rows
    for j = 1:cols
        if(inputV(i,j) ~= 0)
            x = randi(2);
            if(x == 2)
                x = -1;
            end
            outV(i,j) = inputV(i,j) + (0.015 * x);
            if(outV(i,j) > 1)
                outV(i,j) = 1;
            end
            if(outV(i,j) < 0)
                outV(i,j) = 0;
            end 
        end
    end
end

