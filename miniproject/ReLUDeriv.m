function [output] = ReLUDeriv(input)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

if(input > 0)
    output = 1;
else
    output = 0;
end

end

