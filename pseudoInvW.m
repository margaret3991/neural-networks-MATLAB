function [weightM] = pseudoInvW(patternM)
% This function calculates the weight matrix for an autoassociator 
% network using the pseudoinverse rule, W = TP+. 

% Since the pattern and target is the same in an autoassociator network, 
% only one input matrix is needed, which is both the pattern and targets. 

Pplus = inv(patternM' * patternM) * patternM';

weightM = patternM * Pplus;
end

