function [successRate] = flipTest(Weight, patternM, repeats, changePixels)
% This function automates the repeatitive test for the noisy patterns vs 
% the weight matrix calculated from those patterns. 

% Weight = weight matrix being tested
% patternM = matrix holding the patterns (unaltered)
% repeats = number of tests to be run on the weight and noisy patterns
% changePixels = number of flipped values in the pattern matrix to add
%      "noise" that will distort the test input pattern 

% Uses the function addNoise() to flip the selected number of values in the
% input vector held in patternM. Then it uses the perceptron equation to
% get the output. This output is then compared to the original pattern and 
% if it matches, the match counter is incremented. 

matches = 0; count = 0; rep = 0;

[rows, cols] = size(patternM); %grab size 
maxRepeat = ceil(repeats/cols) %max repeat per pattern

while rep <= maxRepeat
    for p = 1:cols
        if count < repeats
            count = count+1 %increment count
            origp = patternM(:,p);
            pflip = addNoise(patternM(:,p), changePixels)
            %now test
            ans = hardlims(Weight * pflip);
            equality = isequal(ans, origp)
            matches = matches + equality
        end
    end
    rep = rep + 1;
end

successRate = (matches/repeats) * 100;
end

