function [percError] = determineAccuracy(testResults,labels)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
[rows, cols] = size(testResults);
numCorrect = sum(testResults == labels)
numIncorrect = rows - numCorrect;
percError = 100 * (numIncorrect/rows);
end

