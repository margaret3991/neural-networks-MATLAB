function err = mse(targets, answers)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
[rows cols] = size(targets);
differences = targets - answers;
differences = differences.^2;

summ = sum(differences);
err = summ/rows;
end

