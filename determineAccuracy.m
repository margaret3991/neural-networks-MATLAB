function [percError] = determineAccuracy(testResults,labels)
%	This function compares the testResults vector against the desired 
%   labels vector to determine how accurate the network is

%   INPUTS:
%   testResults = the vector of network outputs from the validation data
%   labels = the vector of labels for the validation data 
%   OUTPUTS:
%   output = percentage of accurate labels 
[rows, cols] = size(testResults);   % get size of the vectors 
numCorrect = sum(testResults == labels) % sums logical comparison array 
numIncorrect = rows - numCorrect;   % subtract from total number of pairs
percError = 100 * (numIncorrect/rows);  % divide by total * 100 for perc
end

