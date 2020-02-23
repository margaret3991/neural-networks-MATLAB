function [ mse ] = backProp(trainInputs,trainTargets, learningRate, iterations, architecture)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%   trainInputs = input vector for training set 
%   trainTargets = target vector for training set
%   learning rate = alpha learning rate 
%   iterations = set value for maximum iterations 
%   architecture = vector representing number of neurons and layers for the
%       neural network i.e. [2 1] for a 1-2-1 pattern (doesn't count input)

%   transfer function definitions:
%   layer1 function: logsigmoid()
%   layer2 function: linear() 

%   Initial Weights and biases 
W1 = [-0.27; -0.41]
b1 = [-0.48; -0.13]

W2 = [0.09 -0.17]
b2 = [0.48]

input = 1;
target = (1 + sin(pi/4))
alpha = 0.1

%------ Now Propagate Forwards ------%

%product = (W1 * trainInputs + b1)

a1 = logSigmoid(W1 * input + b1)
disp(a1)

a2 = purelin(W2 * a1 + b2)
disp(a2)

error = target - a2;

%------ Now Calculate Sensitivities and Backpropagate ------%

f2n2 = 1
s2 = -2 * f2n2 * error

F1n1 = zeros(2, 2)
[rows cols] = size(F1n1)
for i = 1:rows
    for j = 1:cols
        if i == j
            F1n1(i,j) = (1 - a1(i,1))*(a1(i,1))
        end
    end
end
disp(F1n1)

s1 = F1n1 * W2' * s2

%------ Update Weights and Biases ------%

W2 = W2 - (alpha * s2 * a1')
b2 = b2 - (alpha * s2)

W1 = W1 - (alpha * s1 * input')
b1 = b1 - (alpha * s1)

mse = error^2;
end

